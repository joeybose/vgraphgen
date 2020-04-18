import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def draw_nx_graph(G, name='Lobster', path='./visualization/train_nxgraph/'):
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.set_title(name, fontsize=10)
    nx.draw(G)
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = path + name + '.png'
    plt.savefig(save_name, format="PNG")
    plt.close()

def draw_pyg_graph(G, name='Lobster', path='./visualization/train_pyggraph/'):
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.set_title(name, fontsize=10)
    nx_graph = to_networkx(G)
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = path + name + '.png'
    nx.draw(nx_graph)
    plt.savefig(save_name, format="PNG")
    plt.close()

def draw_graph_list(G_list, row, col, fname='exp/gen_graph.png',
                    layout='spring', is_single=False, k=1, node_size=55,
                    alpha=1, width=1.3):

    os.makedirs(os.path.dirname(fname), exist_ok=True)

    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row, col, i + 1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        # plt.axis("off")

        # turn off axis label
        plt.xticks([])
        plt.yticks([])

        if layout == 'spring':
            pos = nx.spring_layout( G, k=k / np.sqrt(G.number_of_nodes()),
                                   iterations=100)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)

        if is_single:
            # node_size default 60, edge_width default 1.5
            nx.draw_networkx_nodes( G, pos, node_size=node_size,
                                   node_color='#336699', alpha=1, linewidths=0,
                                   font_size=0)
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes( G, pos, node_size=1.5,
                                   node_color='#336699', alpha=1,
                                   linewidths=0.2, font_size=1.5)
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

def draw_graph_list_separate(G_list, fname='exp/gen_graph', layout='spring',
                             is_single=False, k=1, node_size=55, alpha=1,
                             width=1.3):

    for i, G in enumerate(G_list):
        plt.switch_backend('agg')

        plt.axis("off")

        # turn off axis label
        # plt.xticks([])
        # plt.yticks([])

        if layout == 'spring':
            pos = nx.spring_layout( G, k=k / np.sqrt(G.number_of_nodes()),
                                   iterations=100)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)

        if is_single:
            # node_size default 60, edge_width default 1.5
            nx.draw_networkx_nodes( G, pos, node_size=node_size,
                                   node_color='#336699', alpha=1, linewidths=0,
                                   font_size=0)
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes( G, pos, node_size=1.5,
                                   node_color='#336699', alpha=1,
                                   linewidths=0.2, font_size=1.5)
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

    plt.draw()
    plt.tight_layout()
    plt.savefig(fname+'_{:03d}.png'.format(i), dpi=300)
    plt.close()

def gran_vis(args):
    num_col = args.vis_num_row
    num_row = int(np.ceil(args.num_vis / num_col))
    test_epoch = args.dataset
    test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]
    save_gen_base = plots = './visualization/gen_plots/' + args.dataset + '/'
    save_gen_plots = save_gen_base + args.model + str(args.z_dim) + '_' \
        + flow_name + '_' + decoder_name +  '/'
    save_name = os.path.join(save_gen_plots,
                             '{}_gen_graphs_epoch_{}_block_{}_stride_{}.png'.format(args.model,
                                                                                    test_epoch,
                                                                                    args.block_size,
                                                                                    args.stride))

    # remove isolated nodes for better visulization
    graphs_pred_vis = [copy.deepcopy(gg) for gg in graphs_gen[:args.num_vis]]

    if args.better_vis:
        for gg in graphs_pred_vis:
            gg.remove_nodes_from(list(nx.isolates(gg)))

    # display the largest connected component for better visualization
    vis_graphs = []
    for gg in graphs_pred_vis:
        CGs = [gg.subgraph(c) for c in nx.connected_components(gg)]
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
        vis_graphs += [CGs[0]]

    if args.is_single_plot:
        draw_graph_list(vis_graphs, num_row, num_col, fname=save_name, layout='spring')
    else:
        draw_graph_list_separate(vis_graphs, fname=save_name[:-4], is_single=True, layout='spring')

    save_name = os.path.join(save_gen_plots, 'train_graphs.png')

    if args.is_single_plot:
        draw_graph_list(train_loader.dataset[:args.num_vis], num_row, num_col,
                        fname=save_name, layout='spring')
    else:
        draw_graph_list_separate(train_loader.dataset[:args.num_vis],
                                 fname=save_name[:-4], is_single=True,
                                 layout='spring')
