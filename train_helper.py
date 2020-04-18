from test_helper import test_generation
from visualization.utils import *
from utils.utils import pad_adj_mat, get_graph, extract_batch, constraint_mask, get_subgraph
from utils.early_stopping import EarlyStopping
from utils.eval_utils import evaluate_generated
from distributions.normal import EuclideanNormal
from flows.flows import *
from models.model_helpers import calc_mi, calc_au
from test_helper import test_generation, cond_test_generation
import os
import wandb
import torch
import numpy as np
import copy
from torch import optim
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from visualization.utils import draw_nx_graph, draw_pyg_graph
import sys
import networkx as nx

sys.path.append("..")  # Adds higher directory to python modules path.

kwargs_flows = {'MAFRealNVP': MAFRealNVP, 'RealNVP': RealNVP, 'MAF': MAF,
                'MAFMOG':MAFMOG}

def aggressive_burnin(args, epoch, kl_weight, anneal_rate, aggressive_flag,
                      data_batch, model, enc_opt, dec_opt, train_loader,
                      constraints, oracle):
    burn_num_examples = 0
    burn_pre_loss = 1e4
    burn_cur_loss = 0
    sub_iter = 1
    collate_fn = train_loader.collate_fn
    dataset_iter = iter(train_loader.dataset)

    while aggressive_flag and sub_iter < args.sub_iter_limit:
        batch = extract_batch(args, data_batch)
        # Correct shapes for VGAE processing
        if len(batch[0]['adj'].shape) > 2:
            # We give the full Adj to the encoder
            adj = batch[0]['adj'] + batch[0]['adj'].transpose(2, 3)
            node_feats = adj.view(-1, args.max_nodes)
        else:
            node_feats = batch[0]['adj']

        if batch[0]['edges'].shape[0] != 2:
            edge_index = batch[0]['encoder_edges'].long()
        else:
            edge_index = batch[0]['edges']

        enc_opt.zero_grad()
        dec_opt.zero_grad()

        burn_num_examples += batch[0]['adj'].size(0)
        z, z_k = model.encode(node_feats, edge_index)
        batch[0]['node_latents'] = z_k
        batch[0]['return_adj'] = True
        recon_loss, soft_adj_mat = model.decode(batch)

        # compute KL for Real nodes
        # kl_weight = min(1.0, kl_weight + anneal_rate)
        kl_loss = model.kl_loss(z, z_k, free_bits=args.free_bits)
        # Check whether we want to enforce constraints at Train time
        eval_const_cond = constraints is not None and not args.eval_const_test
        if eval_const_cond:
            input_batches = z_k.view(-1, args.max_nodes, args.z_dim)
            domains, max_node_batches = oracle.constraint.get_domains(input_batches)
            if args.decoder == 'gran':
                avg_neg_loss, z_star_batches = oracle.general_pgd(input_batches, domains,
                                                                  max_node_batches, batch, num_restarts=1,
                                                                  num_iters=args.num_iters, args=args)
                batch[0]['node_latents'] = z_star_batches
                _, adj_star = model.decode(batch)
            else:
                constraint_edge_index = constraint_mask(args, model)
                avg_neg_loss, z_star_batches = oracle.general_pgd(input_batches, domains,
                                                                  max_node_batches, constraint_edge_index,
                                                                  num_restarts=1, num_iters=args.num_iters,
                                                                  args=args)
                _, adj_star = model.decoder(z_star_batches, constraint_edge_index, return_adj=True)
            _, constraint_loss, constr_sat = oracle.evaluate(args, max_node_batches, adj_star)
        else:
            constraint_loss = recon_loss * 0
            constr_sat = recon_loss * 0
            avg_neg_loss = recon_loss * 0

        train_loss = recon_loss + kl_weight * kl_loss + args.constraint_weight * constraint_loss
        burn_cur_loss += train_loss.item()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        enc_opt.step()
        ids = np.random.choice(len(train_loader.dataset), args.batch_size, replace=False)
        data_batch = collate_fn([train_loader.dataset[idx] for idx in ids])

        if sub_iter % 10 == 0:
            burn_cur_loss = burn_cur_loss / burn_num_examples
            if burn_pre_loss - burn_cur_loss < 0:
                break
            burn_pre_loss = burn_cur_loss
            burn_cur_loss = burn_num_examples = 0

        sub_iter += 1

    return sub_iter

def do_val(args, kl_weight, model, val_loader, constraints, oracle, epoch):
    model.eval()
    val_loss_avg, recon_loss_avg, kl_avg = [], [], []
    val_loss_avg.append(0)
    recon_loss_avg.append(0)
    kl_avg.append(0)
    for i, data_batch in enumerate(val_loader):
        batch = extract_batch(args, data_batch)
        # Correct shapes for VGAE processing
        if len(batch[0]['adj'].shape) > 2:
            # We give the full Adj to the encoder
            adj = batch[0]['adj'] + batch[0]['adj'].transpose(2, 3)
            node_feats = adj.view(-1, args.max_nodes)
        else:
            node_feats = batch[0]['adj']

        if batch[0]['edges'].shape[0] != 2:
            edge_index = batch[0]['encoder_edges'].long()
        else:
            edge_index = batch[0]['edges']

        if args.model == 'gran':
            train_loss = recon_loss = model(*batch)

            # dummy tensor
            kl_loss = recon_loss * 0
        else:
            z, z_k = model.encode(node_feats, edge_index)
            batch[0]['node_latents'] = z_k
            batch[0]['return_adj'] = True

            recon_loss, soft_adj_mat = model.decode(batch)
            kl_loss = model.kl_loss(z, z_k, free_bits=args.free_bits)

        # Loss calculation
        val_loss = recon_loss + kl_weight * kl_loss
        val_loss_avg[-1] += -1 * val_loss.item()
        recon_loss_avg[-1] += recon_loss.sum(dim=-1).item()
        kl_avg[-1] += kl_loss.item()

    val_loss_avg[-1] /= len(val_loader.dataset)
    recon_loss_avg[-1] /= len(val_loader.dataset)
    kl_avg[-1] /= (i + 1)
    print('Epoch: {:03d}, Recon:{:.4f}, Pseudo KL:{:.6f}, Val Loss: {:.4f},'.format(epoch, recon_loss_avg[-1],
                                      kl_avg[-1], val_loss_avg[-1]))
    model.train()
    return val_loss_avg[-1]

def train_graph_generation(args, train_loader, val_loader, test_loader, model,
                           constraints, oracle):
    # Initialize training diagnostics
    decay_cnt = cur_mi = pre_mi = best_mi = mi_not_improved = 0
    # KL weight increases linearly from KL_start to 1.0 overwarmup epochs
    anneal_rate = (1.0 - args.kl_start) / (args.warmup * len(train_loader.dataset))
    kl_weight = args.kl_start
    aggressive_flag = True if args.aggressive else False
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    if args.model != 'gran':
        enc_opt = optim.Adam(model.encoder.parameters(), lr=args.lr)
        dec_opt = optim.Adam(model.decoder.parameters(), lr=args.lr)
    else:
        enc_opt = None
        dec_opt = optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    ### Start Training ###
    print("Doing KL Anneal with KL start at %f for %d epochs" % (args.kl_start,
                                                                 args.warmup))
    for epoch in range(0, args.epochs):
        constr_sat_avg, train_loss_avg, recon_loss_avg, kl_avg = [], [], [], []
        neg_constr_loss_avg, pos_constr_loss_avg = [], []
        pos_constr_loss_avg.append(0)
        neg_constr_loss_avg.append(0)
        constr_sat_avg.append(0)
        train_loss_avg.append(0)
        recon_loss_avg.append(0)
        kl_avg.append(0)
        for i, data_batch in enumerate(train_loader):
            # Do Burnin
            sub_iter = aggressive_burnin(args, epoch, kl_weight, anneal_rate,
                              aggressive_flag, data_batch, model, enc_opt,
                              dec_opt, train_loader, constraints, oracle)
            if args.aggressive and aggressive_flag:
                print("Done Aggressive Burn for Epoch %d in %d iters"
                      % (epoch, sub_iter))
                enc_opt.zero_grad()
                dec_opt.zero_grad()

            batch = extract_batch(args, data_batch)
            # Correct shapes for VGAE processing
            if len(batch[0]['adj'].shape) > 2:
                # We give the full Adj to the encoder
                adj = batch[0]['adj'] + batch[0]['adj'].transpose(2, 3)
                node_feats = adj.view(-1, args.max_nodes)
            else:
                node_feats = batch[0]['adj']

            if batch[0]['edges'].shape[0] != 2:
                edge_index = batch[0]['encoder_edges'].long()
            else:
                edge_index = batch[0]['edges']

            if args.model == 'gran':
                train_loss = recon_loss = model(*batch)

                # dummy tensor
                kl_loss = recon_loss * 0
            else:
                z, z_k = model.encode(node_feats, edge_index)
                batch[0]['node_latents'] = z_k
                batch[0]['return_adj'] = True

                recon_loss, soft_adj_mat = model.decode(batch)
                kl_weight = min(1.0, kl_weight + anneal_rate)
                kl_loss = model.kl_loss(z, z_k, free_bits=args.free_bits)

            # Check whether we want to enforce constraints at Train time
            eval_const_cond = constraints is not None and not args.eval_const_test
            if eval_const_cond:
                input_batches = z_k.view(-1, args.max_nodes, args.z_dim)
                domains, max_node_batches = oracle.constraint.get_domains(input_batches)
                if args.decoder == 'gran':
                    avg_neg_loss, z_star_batches = oracle.general_pgd(input_batches, domains,
                                                                      max_node_batches, batch, num_restarts=1,
                                                                      num_iters=args.num_iters, args=args)
                    batch[0]['node_latents'] = z_star_batches
                    _, adj_star = model.decode(batch)
                else:
                    constraint_edge_index = constraint_mask(args, model)
                    avg_neg_loss, z_star_batches = oracle.general_pgd(input_batches, domains,
                                                                      max_node_batches, constraint_edge_index,
                                                                      num_restarts=1, num_iters=args.num_iters,
                                                                      args=args)
                    _, adj_star = model.decoder(z_star_batches, constraint_edge_index, return_adj=True)
                _, constraint_loss, constr_sat = oracle.evaluate(args, max_node_batches, adj_star)
            else:
                constraint_loss = recon_loss * 0
                constr_sat = recon_loss * 0
                avg_neg_loss = recon_loss * 0

            # Loss calculation
            train_loss = recon_loss + kl_weight * kl_loss + args.constraint_weight * constraint_loss
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            if not aggressive_flag and args.model != 'gran':
                enc_opt.step()

            dec_opt.step()

            # Online Logging
            train_loss_avg[-1] += -1 * train_loss.item()
            recon_loss_avg[-1] += recon_loss.sum(dim=-1).item()
            kl_avg[-1] += kl_loss.item()
            print("KL Loss %f Epoch: %d" %(kl_loss.item(), epoch))
            pos_constr_loss_avg[-1] += constraint_loss.item()
            neg_constr_loss_avg[-1] += avg_neg_loss.item()
            constr_sat_avg[-1] += constr_sat

            train_loss_metric = 'Train Loss'
            elbo_metric = 'Elbo'
            recon_loss_metric = 'Reconstruction'
            kl_metric = 'Pseudo KL-Divergence'

            if args.wandb:
                wandb.log({train_loss_metric: train_loss.item(),
                           elbo_metric: -1 * train_loss.item(),
                           recon_loss_metric: recon_loss.item(),
                           kl_metric: kl_loss.item(), "x": epoch,
                           'Pos Cons. Loss': constraint_loss.item(),
                           'dataset': args.dataset,
                           })

            if (i + 1) % 10 == 0:
                if aggressive_flag:
                    model.eval()
                    with torch.no_grad():
                        mi = calc_mi(args, model, test_loader)
                        # au, _ = calc_au(model, val_loader)
                    model.train()
                print(
                    'Train Loss : {:.4f}, MI: {:.4f}'.format(train_loss_avg[-1] / i, mi)) if aggressive_flag else print(
                    'Train Loss : {:.4f}'.format(train_loss_avg[-1] / i))

        # Reset Burnin Flags after each Epoch
        if aggressive_flag:
            model.eval()
            cur_mi = calc_mi(args, model, test_loader)
            model.train()
            if cur_mi - best_mi < 0:
                mi_not_improved += 1
                # if mi_not_improved == 5:
                    # aggressive_flag = False
                    # print("STOP BURNING")
            else:
                best_mi = cur_mi

            pre_mi = cur_mi

        train_loss_avg[-1] /= len(train_loader.dataset)
        recon_loss_avg[-1] /= len(train_loader.dataset)
        kl_avg[-1] /= (i + 1)
        neg_constr_loss_avg[-1] /= (i + 1)
        pos_constr_loss_avg[-1] /= (i + 1)
        constr_sat_avg[-1] /= (i + 1)
        print('Epoch: {:03d}, Recon:{:.4f}, Pseudo KL:{:.6f}, Train Loss: {:.4f}, '
              'Neg Constr Loss: {:.4f}, Pos Constr Loss: {:.4f}, '
              'Constr Sat: {:.4f}'.format(epoch, recon_loss_avg[-1],
                                          kl_avg[-1], train_loss_avg[-1], neg_constr_loss_avg[-1],
                                          pos_constr_loss_avg[-1], constr_sat_avg[-1]))

        val_loss = do_val(args, kl_weight, model, val_loader, constraints,
                          oracle, epoch)
        early_stopping(-1*val_loss, model)

        if (epoch + 1) % args.sample_every == 0:
            # TODO: memory leak?
            if args.model == 'gran':
                test_generation(args, train_loader, test_loader, model, 'gran',
                                None, epoch=epoch, oracle=oracle)
            else:
                test_generation(args, train_loader, test_loader, model.decoder,
                        model.decoder_name, model.encoder.flow_model,
                        epoch=epoch, oracle=oracle)
                cond_test_generation(args, train_loader, test_loader, model,
                                     model.decoder, model.decoder_name,
                                     model.encoder.flow_model, epoch=epoch,
                                     oracle=oracle)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model
