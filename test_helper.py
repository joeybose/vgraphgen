import os
import wandb
import torch
import numpy as np
import copy
from torch import optim
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from visualization.utils import draw_nx_graph, draw_pyg_graph, draw_graph_list
from utils.utils import pad_adj_mat, get_graph, extract_batch, constraint_mask, get_subgraph
from utils.eval_utils import evaluate_generated
import sys
import networkx as nx
import ipdb

def test_generation(args, train_loader, test_loader, decoder, decoder_name,
        flow_model=None, epoch=-1, oracle=None):

    node_dist = args.node_dist
    edge_index = None
    flow_name = ''

    if args.flow_model:
        flow_name = args.flow_model

    if not decoder_name:
        decoder_name = ''

    save_gen_base = plots = './visualization/gen_plots/' + args.dataset + '/'
    save_gen_plots = save_gen_base + args.model + str(args.z_dim) + '_' \
        + str(flow_name) + '_' + decoder_name + '/'
    gen_graph_list, gen_graph_copy_list = [], []
    avg_connected_components, avg_triangles, avg_transitivity = [], [], []
    raw_triangles = []

    if args.model == 'gran':
        num_nodes_pmf = train_loader.dataset.num_nodes_pmf
        model = decoder
        model.eval()
        A = decoder._sampling(args.num_gen_samples)
        num_nodes_pmf = torch.from_numpy(num_nodes_pmf).to(model.device)
        num_nodes = torch.multinomial(num_nodes_pmf.float(),
                args.num_gen_samples, replacement=True)  # shape B X 1

        A_list = [ A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in
                  range(args.num_gen_samples) ]
        adj_mats_padded = pad_adj_mat(args, A_list)
        model.train()
    else:  # VAE
        # sample lengths of graphs
        if args.decoder == 'gran':
            num_nodes_pmf = train_loader.dataset.num_nodes_pmf
            num_nodes_pmf = torch.from_numpy(num_nodes_pmf).to(args.dev)
            num_nodes = torch.multinomial(num_nodes_pmf.float(),
                    args.num_gen_samples, replacement=True)  # shape B X 1
        else:
            num_nodes_pmf = np.random.choice(node_dist, args.num_gen_samples)

        batch_all_nodes = []
        for i, len_ in enumerate(num_nodes):
            fully_connected = nx.complete_graph(len_)
            edge_index_i = torch.tensor(list(fully_connected.edges)).to(args.dev).t().contiguous()
            batch_all_nodes += [edge_index_i + i * args.max_nodes]

        edge_index = torch.cat(batch_all_nodes, 1).to(args.dev)

        if flow_model is None:
            z_0 = torch.FloatTensor( args.num_gen_samples * args.max_nodes,
                                    args.z_dim).to(args.dev).normal_()
            z_k = z_0
        else:
            z_0 = flow_model.base_dist.sample((args.num_gen_samples,
                                               flow_model.n_components)).squeeze()
            # z_0 = z_0.view(args.num_gen_samples, -1)
            z_k, _ = flow_model.inverse(z_0, edge_index=edge_index)
            z_k = z_k.view(args.num_gen_samples * args.max_nodes, args.z_dim)

        if args.decoder == 'gran':
            decoder.eval()
            A = decoder._sampling(
                args.num_gen_samples, enc_node_feats=z_k)

            A_list = [ A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in
                      range(args.num_gen_samples) ]
            # This is only needed for constrain sat eval, padded rows will be
            # masked out again. We have to check for 0-rows before Max Nodes
            # though which is why we cant just not-pad.
            adj_mats_padded = pad_adj_mat(args, A_list)
            decoder.train()
        else:
            num_nodes = None
            decoder.eval()
            adj_mats = decoder(z_k, edge_index, return_adj=True)[-1]
            decoder.train()
            if args.deterministic_decoding:
                adj_mats = (adj_mats > 0.5).float()
            else:
                adj_mats = torch.bernoulli(adj_mats)

    for adj_mat in A_list:
        g = nx.from_numpy_matrix(adj_mat.detach().cpu().numpy())
        g.remove_edges_from(nx.selfloop_edges(g))
        g_copy = copy.deepcopy(g)
        gen_graph_copy_list.append(g_copy)

        if len(g) > 0:
            # process the graphs
            if args.better_vis:
                g = max(nx.connected_component_subgraphs(g), key=len)
            num_connected_components = nx.number_connected_components(g)
            avg_connected_components.append(num_connected_components)
            num_triangles = list(nx.triangles(g).values())
            avg_triangles.append(sum(num_triangles) /
                                 float(len(num_triangles)))
            avg_transitivity.append(nx.transitivity(g))
            raw_triangles.append([num_triangles, len(g.nodes)])
            gen_graph_list.append(g)

    # once graphs are generated
    total = len(gen_graph_list)  # min(3, len(vis_graphs))
    if args.constraint:
        constraint_str = args.constraint
        _, constraint_loss, constr_sat = oracle.evaluate(args, num_nodes,
                                                         adj_mats_padded)
    else:
        constraint_str = ''
        constraint_loss, constr_sat = 0, 0

    draw_graph_list(gen_graph_list[:total], 3, int(total // 3),
                    fname='./visualization/sample/{}/{}_{}.png'.format(args.namestr,
                                                                       constraint_str,
                                                                       epoch),
                    layout='spring')

    # Evaluate Generated Graphs using GraphRNN metrics
    if args.decoder == 'gran' or args.model == 'gran':
        test_dataset = [test_G for test_G in test_loader.dataset.graphs]
    else:
        test_dataset = [to_networkx(test_G).to_undirected()
                        for test_G in test_loader]
    metrics = evaluate_generated(
        test_dataset, gen_graph_list, args.dataset)
    metrics_copy = evaluate_generated(
        test_dataset, gen_graph_copy_list, args.dataset)
    # Orginal Graphs with nodes remoed
    mmd_degree, mmd_clustering, mmd_4orbits = metrics[0], metrics[1], metrics[2]
    mmd_spectral, accuracy = metrics[3], metrics[4]
    mean_connected_comps = sum(
        avg_connected_components) / float(len(avg_connected_components))
    mean_triangles = sum(avg_triangles) / float(len(avg_triangles))
    mean_transitivity = sum(avg_transitivity) / \
        float(len(avg_transitivity))

    # Copied Graphs with nodes not removed
    mmd_degree_copy, mmd_clustering_copy, mmd_4orbits_copy = metrics_copy[
        0], metrics_copy[1], metrics_copy[2]
    mmd_spectral_copy, accuracy_copy = metrics_copy[3], metrics_copy[4]
    if args.wandb:
        wandb.log({"Deg": mmd_degree, "Clus": mmd_clustering, "Orb":
                   mmd_4orbits, "Acc": accuracy, "Spec.": mmd_spectral,
                   "Avg_CC": mean_connected_comps, "Avg_Tri": mean_triangles,
                   "Avg_transitivity": mean_transitivity, "Raw_triangles":
                   raw_triangles, "Deg_copy": mmd_degree_copy, "Clus_copy":
                   mmd_clustering_copy, "Orb_copy": mmd_4orbits_copy,
                   "Acc_copy": accuracy_copy, "Spec_copy": mmd_spectral_copy,
                   "Test Constr Loss": constraint_loss, "Test Constr Sat":
                   constr_sat, "test_step": epoch})

    print('Deg: {:.4f}, Clus.: {:.4f}, Orbit: {:.4f}, Spec.:{:.4f}, Acc: {:.4f}'.format(mmd_degree,
                                                                                        mmd_clustering,
                                                                                        mmd_4orbits,
                                                                                        mmd_spectral,
                                                                                        accuracy))
    print('Avg CC: {:.4f}, Avg. Tri: {:.4f}, Avg. Trans: {:.4f}'.format(mean_connected_comps, mean_triangles,
                                                                        mean_transitivity))
    print('Test Constr Loss: {:.4f}, Test Constr Sat: {:.4f}'.format(constraint_loss, constr_sat))
    return [mmd_degree, mmd_clustering, mmd_4orbits, mmd_spectral, accuracy]

def cond_test_generation(args, train_loader, test_loader, model, decoder, decoder_name,
        flow_model=None, epoch=-1, oracle=None):

    node_dist = args.node_dist
    edge_index = None
    flow_name = ''
    model.eval()
    if args.flow_model:
        flow_name = args.flow_model

    if not decoder_name:
        decoder_name = ''

    save_gen_base = plots = './visualization/gen_plots/' + args.dataset + '/'
    save_gen_plots = save_gen_base + args.model + str(args.z_dim) + '_' \
        + str(flow_name) + '_' + decoder_name + '/'
    gen_graph_list, gen_graph_copy_list = [], []
    avg_connected_components, avg_triangles, avg_transitivity = [], [], []
    raw_triangles = []
    A_list = []
    test_recon_loss_avg = []
    test_recon_loss_avg.append(0)
    for i, data_batch in enumerate(test_loader):
        batch = extract_batch(args, data_batch)
        # Correct shapes for VGAE processing
        if len(batch[0]['adj'].shape) > 2:
            # We give the full Adj to the encoder
            adj = batch[0]['adj'] + batch[0]['adj'].transpose(2,3)
            node_feats = adj.view(-1, args.max_nodes)
        else:
            node_feats = batch[0]['adj']

        if batch[0]['edges'].shape[0] != 2:
            edge_index = batch[0]['encoder_edges'].long()
        else:
            edge_index = batch[0]['edges']

        z, z_k = model.encode(node_feats, edge_index)
        batch[0]['node_latents'] = z_k
        test_recon_loss = model.decode(batch)
        test_recon_loss_avg[-1] += test_recon_loss.sum(dim=-1).item()

        if args.decoder == 'gran':
            decoder.eval()
            num_nodes_pmf = train_loader.dataset.num_nodes_pmf
            num_adj_batch = batch[0]['adj'].size(0)
            A = decoder._sampling(num_adj_batch, enc_node_feats=z_k)
            A_list += [A[ii, :batch[0]['num_nodes_gt'][ii],
                         :batch[0]['num_nodes_gt'][ii]] for ii in
                       range(num_adj_batch)]
            # This is only needed for constrain sat eval, padded rows will be
            # masked out again. We have to check for 0-rows before Max Nodes
            # though which is why we cant just not-pad.
            adj_mats_padded = pad_adj_mat(args, A_list)
            decoder.train()
        else:
            num_nodes = None
            decoder.eval()
            adj_mats = decoder(z_k, edge_index, return_adj=True)[-1]
            decoder.train()
            if args.deterministic_decoding:
                adj_mats = (adj_mats > 0.5).float()
            else:
                adj_mats = torch.bernoulli(adj_mats)

    num_nodes = []
    for adj_mat in A_list:
        g = nx.from_numpy_matrix(adj_mat.detach().cpu().numpy())
        g.remove_edges_from(nx.selfloop_edges(g))
        g_copy = copy.deepcopy(g)
        gen_graph_copy_list.append(g_copy)
        num_nodes.append(g.number_of_nodes())

        if len(g) > 0:
            # process the graphs
            if args.better_vis:
                g = max(nx.connected_component_subgraphs(g), key=len)
            num_connected_components = nx.number_connected_components(g)
            avg_connected_components.append(num_connected_components)
            num_triangles = list(nx.triangles(g).values())
            avg_triangles.append(sum(num_triangles) /
                                 float(len(num_triangles)))
            avg_transitivity.append(nx.transitivity(g))
            raw_triangles.append([num_triangles, len(g.nodes)])
            gen_graph_list.append(g)

    # once graphs are generated
    model.train()
    total = len(gen_graph_list)  # min(3, len(vis_graphs))
    if args.constraint:
        constraint_str = args.constraint
        _, constraint_loss, constr_sat = oracle.evaluate(args, num_nodes,
                                                         adj_mats_padded)
    else:
        constraint_str = ''
        constraint_loss, constr_sat = 0, 0

    draw_graph_list(gen_graph_list[:args.num_gen_samples], 3, int(total // 3),
                    fname='./visualization/sample/{}/Cond_{}_{}.png'.format(args.namestr,
                                                                       constraint_str,
                                                                       epoch),
                    layout='spring')

    # Evaluate Generated Graphs using GraphRNN metrics
    if args.decoder == 'gran' or args.model == 'gran':
        test_dataset = [test_G for test_G in test_loader.dataset.graphs]
    else:
        test_dataset = [to_networkx(test_G).to_undirected()
                        for test_G in test_loader]
    metrics = evaluate_generated(
        test_dataset, gen_graph_list, args.dataset)
    metrics_copy = evaluate_generated(
        test_dataset, gen_graph_copy_list, args.dataset)
    # Orginal Graphs with nodes remoed
    mmd_degree, mmd_clustering, mmd_4orbits = metrics[0], metrics[1], metrics[2]
    mmd_spectral, accuracy = metrics[3], metrics[4]
    mean_connected_comps = sum(
        avg_connected_components) / float(len(avg_connected_components))
    mean_triangles = sum(avg_triangles) / float(len(avg_triangles))
    mean_transitivity = sum(avg_transitivity) / \
        float(len(avg_transitivity))

    # Copied Graphs with nodes not removed
    mmd_degree_copy, mmd_clustering_copy, mmd_4orbits_copy = metrics_copy[
        0], metrics_copy[1], metrics_copy[2]
    mmd_spectral_copy, accuracy_copy = metrics_copy[3], metrics_copy[4]
    test_recon_loss_avg[-1] /= len(test_loader.dataset)
    if args.wandb:
        wandb.log({"Cond Deg": mmd_degree, "Cond Clus": mmd_clustering,
                   "Cond Orb": mmd_4orbits, "Cond Acc": accuracy, "Cond Spec.":
                   mmd_spectral, "Cond Avg_CC": mean_connected_comps,
                   "Cond Avg_Tri": mean_triangles, "Cond Avg_transitivity":
                   mean_transitivity, "Cond Raw_triangles": raw_triangles,
                   "Cond Deg_copy": mmd_degree_copy, "Cond Clus_copy":
                   mmd_clustering_copy, "Cond Orb_copy": mmd_4orbits_copy,
                   "Cond Acc_copy": accuracy_copy, "Cond Spec_copy":
                   mmd_spectral_copy, "Cond Test Constr Loss": constraint_loss,
                   "Cond Test Constr Sat": constr_sat, "Test Recon Loss":
                   test_recon_loss_avg[-1], "test_step": epoch})

    print('Cond. Deg: {:.4f}, Clus.: {:.4f}, Orbit: {:.4f}, Spec.:{:.4f}, Acc: {:.4f}'.format(mmd_degree,
                                                                                        mmd_clustering,
                                                                                        mmd_4orbits,
                                                                                        mmd_spectral,
                                                                                        accuracy))
    print('Cond. Avg CC: {:.4f}, Avg. Tri: {:.4f}, Avg. Trans: {:.4f}'.format(mean_connected_comps, mean_triangles,
                                                                        mean_transitivity))
    print('Cond. Test Constr Loss: {:.4f}, Cond. Test Constr Sat: {:.4f}'.format(constraint_loss, constr_sat))
    return [mmd_degree, mmd_clustering, mmd_4orbits, mmd_spectral, accuracy]
