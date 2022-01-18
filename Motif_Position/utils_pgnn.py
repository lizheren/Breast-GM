import copy
import math
import random
import numpy as np
import torch
from torch import tensor
import networkx as nx
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import init

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def single_source_shortest_path_length_range(graph, node_range):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node,cutoff=2)
        # Compute the shortest path lengths from source to all reachable nodes.
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def all_pairs_shortest_path_length_parallel(graph,  num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    results = [single_source_shortest_path_length_range(graph, nodes)]
    dists_dict = merge_dicts(results)
    # print(dists_dict)
    return dists_dict


def precompute_dist_data(graph, num_nodes, approximate=1):

    n = num_nodes
    dists_array = np.zeros((n, n))
    # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
    # dists_dict = {c[0]: c[1] for c in dists_dict}
    dists_dict = all_pairs_shortest_path_length_parallel(graph)
    for i, node_i in enumerate(graph.nodes()):

        shortest_dist = dists_dict[node_i]
        dist_k = []
        for k, node_k in enumerate(graph.nodes()):
            distance = shortest_dist.get(node_k, -1)
            if distance != -1:
                dist_k.append(distance)
        # dist_sum = sum(dist_k)
        dist_max = max(dist_k)
        # dist_mean = np.mean(dist_k)
        m = 0
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)

            if dist != -1:
                # dists_array[i, j] = 1 / (dist + 1)
                # att_dist = dist_k[m]/dist_mean
                # att_dist = abs((dist_k[m]-dist_mean)/dist_mean)
                #att_dist = dist_k[m] / dist_max
                dists_array[node_i, node_j] = 1 / (dist + 1)
                #dists_array[node_i, node_j] = math.exp(-(dist * att_dist))
                m += 1
    # print(dists_array)
    return dists_array


def get_dist_max(anchorset_id, dist):
    dist_max = torch.zeros((dist.shape[0], len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0], len(anchorset_id))).long().to(device)
    #print(anchorset_id)
    for i in range(len(anchorset_id)):
        temp_id = torch.as_tensor(anchorset_id[i], dtype=torch.long)
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:, i] = dist_max_temp
        #print(temp_id,"\n", dist_argmax[:,0],"\n",dist_max,dist_argmax_temp)
        dist_argmax[:, i] = temp_id
    # print(dist_max, "\t", dist_argmax)
    return dist_max, dist_argmax

'''
def get_random_anchorset(n, c=0.5):
    m = int(np.log2(n))
    #m = int(np.log(n))
    # print(m)
    copy = int(c * m)
    # print(copy)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n / np.exp2(i + 1))
        #anchor_size = int(n / np.exp(i + 1))
        for j in range(copy-1):
            anchorset_id.append(np.random.choice(n, size=anchor_size, replace=False))
    print(anchorset_id)
    resume2 =False
    assert resume2,"break"
    return anchorset_id
'''
def get_random_anchorset(n, c=0.5):
    m = int(np.log2(n))
    #m = int(np.log(n))
    # print(m)
    copy = int(c * m)
    # print(copy)
    anchorset_id = []
    # for i in range(m):
    #     anchor_size = int(n / np.exp2(i + 1))
    #     #anchor_size = int(n / np.exp(i + 1))
    #     for j in range(copy-1):
    #         anchorset_id.append(np.random.choice(n, size=anchor_size, replace=False))
    # print(anchorset_id)
    for i in range(n):
        anchorset_id.append(np.array(i))
    #print(anchorset_id)
    #resume2 = False
    #assert resume2,"break"
    return anchorset_id


def preselect_anchor(graph, layer_num=1, anchor_num=6, anchor_size_num=4, device='cpu'):
    anchor_size_num = anchor_size_num
    anchor_set = []
    anchor_num_per_size = anchor_num // anchor_size_num
    for i in range(anchor_size_num):
        anchor_size = 2 ** (i + 1) - 1
        # anchor_size = int(math.exp(i + 1) - 1)
        anchors = np.random.choice(len(graph), size=(layer_num, anchor_num_per_size, anchor_size),
                                   replace=True)
        anchor_set.append(anchors)
    anchor_set_indicator = np.zeros((layer_num, anchor_num, len(graph)), dtype=int)

    anchorset_id = get_random_anchorset(len(graph), c=1)
    # print(anchorset_id)
    #resume2 =False
    #assert resume2,"break"
    return anchorset_id


def convert(tensor, length):

    graph = []
    G = nx.Graph()
    for i in range(length):
        graph.append(G)
    copy_graph = []
    for k in range(length):

        graph_np = tensor[k].cpu().numpy()
        graph_int = np.squeeze(graph_np.astype(int))

        graph[k] = nx.from_numpy_matrix(graph_int)
        copy_graph.append(copy.deepcopy(graph[k]))
        # print(graph[k].edges())
        # print(np.array(nx.adjacency_matrix(graph[k]).todense()))
        graph[k].clear()

    return copy_graph


def caculate(graphs):
    Graph = convert(graphs, len(graphs))
    dists_max = [[]]*len(graphs)
    dists_argmax = [[]]*len(graphs)
    dist_graph = []
    for i in range(len(graphs)):
        dist = precompute_dist_data(Graph[i], Graph[i].number_of_nodes())
        anchid = preselect_anchor(Graph[i])
        dist_graph.append(torch.from_numpy(dist))
        dists_max[i], dists_argmax[i] = get_dist_max(anchid, torch.from_numpy(dist).float())
    return dists_max,dists_argmax,dist_graph
