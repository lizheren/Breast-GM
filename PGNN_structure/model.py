#import utils
import torch

from utils.sinkhorn import Sinkhorn
from utils.voting_layer import Voting
from Motif_Position.displacement_layer import Displacement
from utils.feature_align import feature_align
from PGNN_structure.gconv import Siamese_Gconv
from PGNN_structure.affinity_layer import Affinity

from utils.config import cfg
import utils.backbone as bc
from Motif_Position.test_pgnn import *

Nm = 8
SCALE = 4  # 6
BASE = 2
OVERLAP_THRESH = 2  # 1
MIN_MOTIFS = 5  # 70
MOTIF_COUNT_THRESH = 3e2  # 1e4

CNN = eval('bc.{}'.format(cfg.BACKBONE))

class gen_hyper_graph2:

    def __init__(self):
        # self.tree_deg_thresh = 3
        # self.path_len_thresh = 4

        self.paths = []
        self.trees = []
        self.circuits = []

        self.paths_set = []
        self.trees_set = []
        self.circuits_set = []

        self.paths_label = []
        self.trees_label = []
        self.circuits_label = []

        self.data_path = r'GIN_dataset'  # r'D:\QGNN\data'
        self.label_path = r'dataset'
        self.output_path = r'dataset'
        self.datasets = ['PROTEINS']  # 'PTC_MR','MUTAG','NCI1','IMDB-BINARY','IMDB-MULTI']
        self.node_label_count = {'AIDS': 37, 'COLLAB': 0, 'IMDB-BINARY': 0,
                                 'IMDB-MULTI': 0, 'MUTAG': 7, 'NCI1': 37,
                                 'PROTEINS_full': 61, 'PTC_FM': 18, 'PTC_FR': 19,
                                 'PTC_MM': 20, 'PTC_MR': 18}

        self.pass_num = [207800]

        self.A = np.zeros((7, 7))
        edges = [[0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [5, 6]]
        for e in edges:
            self.A[e[0]][e[1]] = 1
            self.A[e[1]][e[0]] = 1

        self.MAX_DEPTH = 10
        self.depth = 0
        self.is_cluster = False

    def run(self):

        # gen generated graph
        motif_labels = []
        motif2A = []

        hyperA, motif2A, motif_label = self.gen_hyperA()  # motif2A

        if motif2A.shape[0] == 0 or motif2A.shape[0] == 1:
            hyperA = self.A.copy()
            motif_label = np.array([[1, 4] for As in range(self.A.shape[0])])
            motif2A = np.eye(self.A.shape[0])
            # print(self.A.shape[0])
        motif_labels.append(motif_label)

        return hyperA, motif2A, motif_label

    def gen_hyperA(self, init_nodeid=0):
        self.count_motif2()
        '''
        #self.count_motif(init_nodeid, [],[])
        if len(self.trees)+len(self.paths)+len(self.circuits)==0:
            for i in range(self.A.shape[0]):
                self.count_motif(i, [],[])
                if len(self.trees) + len(self.paths) + len(self.circuits) != 0:
                    break
        '''
        motifs = np.array(self.trees + self.paths + self.circuits)
        motifs_label = self.trees_label + self.paths_label + self.circuits_label
        self.trees, self.paths, self.circuits = [], [], []
        self.trees_label, self.paths_label, self.circuits_label = [], [], []
        self.paths_set, self.trees_set, self.circuits_set = [], [], []

        motifA = np.zeros((len(motifs), len(motifs)))
        same_motifs = []
        for i1 in range(len(motifs)):
            for i2 in range(i1 + 1, len(motifs)):

                common = [x for x in motifs[i1] if x in motifs[i2]]

                # two motifs are highly overlap or same
                if motifs_label[i1] == motifs_label[i2] and (len(motifs) - len(same_motifs)) > MIN_MOTIFS:
                    if np.array(motifs[i1]).shape[0] - len(common) <= OVERLAP_THRESH and i1 not in same_motifs:
                        same_motifs.append(i1)
                        break
                    elif np.array(motifs[i2]).shape[0] - len(common) <= OVERLAP_THRESH and i2 not in same_motifs:
                        same_motifs.append(i2)
                        continue

                if np.array(motifs[i1]).shape[0] == np.array(motifs[i2]).shape[0]:
                    # same kind but different motifs have common nodes
                    if len(common) != 0:
                        motifA[i1][i2] = 2
                        motifA[i2][i1] = 2

                else:
                    # different kinds and different motifs have common nodes
                    if len(common) != 0:
                        motifA[i1][i2] = 3
                        motifA[i2][i1] = 3

                if len(common) == 0:
                    for j1 in motifs[i1]:
                        for j2 in motifs[i2]:
                            if self.A[j1][j2] == 1:
                                motifA[i1][i2] = 1
                                break
                        if motifA[i1][i2] == 1:
                            break

        motifA = np.delete(motifA, same_motifs, axis=0)
        motifA = np.delete(motifA, same_motifs, axis=1)

        motifs = np.array(motifs)
        motifs = np.delete(motifs, same_motifs, axis=0)
        if motifs.shape[0] > MOTIF_COUNT_THRESH:
            print(motifs.shape[0])

        motifs_label = np.array(motifs_label)
        motifs_label = np.delete(motifs_label, same_motifs, axis=0)

        motif2A = np.zeros((motifs.shape[0], self.A.shape[0]))
        for i in range(motifs.shape[0]):
            motif = motifs[i]
            for node in motif:
                motif2A[i][node] = 1

        return motifA, motif2A, motifs_label

    # non-recursion count method
    def count_motif2(self):

        self.paths = []
        self.trees = []
        self.circuits = []

        self.node2neis = []
        self.neis2node = []

        self.max_hop = [0 for i in range(self.A.shape[0])]

        # find 1-hop neighbours
        for node_id in range(self.A.shape[0]):
            line = self.A[node_id]
            neighbors = set(np.where(line > 0)[0])
            if len(neighbors) != 0:
                self.max_hop[node_id] = 1
            self.node2neis.append([neighbors, neighbors.copy()])
            self.neis2node.append([set() for i in range(self.A.shape[0])])

            # if tree existed
            tree_deg_thresh = 3
            if len(neighbors) + 1 > tree_deg_thresh:
                neighbors.add(node_id)
                tree_set = neighbors
                if tree_set not in self.trees_set and len(self.trees) <= MOTIF_COUNT_THRESH:
                    self.trees_set.append(tree_set)
                    self.trees.append(list(tree_set))
                    self.trees_label.append([len(list(tree_set)), 2])

        # find 2-hop neighbours
        for node_id in range(self.A.shape[0]):
            neis2 = set()
            for nei_id in self.node2neis[node_id][1]:
                neis2_set = (self.node2neis[nei_id][1] - {node_id})
                if len(neis2_set) != 0:
                    self.max_hop[node_id] = 2
                neis2 = neis2 | neis2_set
                for ns in neis2_set:

                    # if 3-nodes circuit existing
                    if ns in self.node2neis[node_id][1]:
                        if {node_id, ns, nei_id} not in self.circuits_set:
                            self.circuits.append([node_id, ns, nei_id])
                            self.circuits_set.append({node_id, ns, nei_id})
                            self.circuits_label.append([3, 3])
                        continue

                    self.neis2node[node_id][ns].add(nei_id)
            self.node2neis[node_id].append(neis2)
            self.node2neis[node_id][0] = self.node2neis[node_id][0] | neis2

        self.find_pown_hop(SCALE)

        # find paths
        path_len_thresh = 4
        for node_id in range(self.A.shape[0]):
            max_hop = self.max_hop[node_id]
            if max_hop + 2 <= path_len_thresh or len(self.paths) >= MOTIF_COUNT_THRESH:
                continue
            for nei_id in self.node2neis[node_id][max_hop]:
                path_set = self.neis2node[node_id][nei_id] | {node_id, nei_id}
                if path_set not in self.paths_set:
                    self.paths_set.append(path_set)
                    self.paths.append(list(path_set))
                    self.paths_label.append([len(list(path_set)), 1])

        return

    def find_pown_hop(self, n):
        for j in range(1, n):
            powj = pow(BASE, j)
            powj1 = pow(BASE, j + 1)
            for node_id in range(self.A.shape[0]):
                existing_neis = set()
                neis = [set() for i in range(powj1 - powj)]
                for i in range(powj + 1, powj1 + 1):
                    for nei_id in self.node2neis[node_id][powj]:
                        neis_set = (self.node2neis[nei_id][i - powj] - {node_id})
                        neis_set_new = set()
                        for ns in neis_set:

                            # back path
                            if ns in existing_neis or ns in self.neis2node[node_id][nei_id] or \
                                    len(self.neis2node[node_id][ns] & self.neis2node[node_id][nei_id]) != 0 or \
                                    len(self.neis2node[node_id][ns] & self.neis2node[nei_id][ns]) != 0 or \
                                    len(self.neis2node[node_id][nei_id] & self.neis2node[nei_id][ns]) != 0:
                                continue

                            # if circuit existed
                            if len(self.neis2node[node_id][ns]) != 0 and ns not in self.neis2node[node_id][nei_id]:
                                circuit_set = self.neis2node[node_id][ns] | self.neis2node[ns][nei_id] | \
                                              self.neis2node[node_id][nei_id] | {node_id, nei_id, ns}
                                if circuit_set not in self.circuits_set and len(self.circuits) <= MOTIF_COUNT_THRESH:
                                    self.circuits_set.append(circuit_set)
                                    self.circuits.append(list(circuit_set))
                                    self.circuits_label.append([len(list(circuit_set)), 3])
                                continue
                            '''
                            if ns in existing_neis or len(self.neis2node[node_id][ns]) != 0 or len(
                                    self.neis2node[nei_id][ns] & self.neis2node[node_id][nei_id]) != 0:
                                continue
                            '''
                            neis_set_new.add(ns)
                            self.neis2node[node_id][ns] = self.neis2node[node_id][ns].union(
                                self.neis2node[node_id][nei_id] | self.neis2node[nei_id][ns])
                            self.neis2node[node_id][ns].add(nei_id)

                        neis[i - powj - 1] = neis[i - powj - 1] | neis_set_new
                        existing_neis = existing_neis | neis[i - powj - 1]

                    if len(neis[i - powj - 1]) != 0:
                        self.max_hop[node_id] += 1
                    self.node2neis[node_id].append(neis[i - powj - 1])
                    self.node2neis[node_id][0] = self.node2neis[node_id][0] | existing_neis


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.bi_stochastic = Sinkhorn(max_iter=cfg.PGNN_structure.BS_ITER_NUM, epsilon=cfg.PGNN_structure.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.PGNN_structure.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.PGNN_structure.FEATURE_CHANNEL * 2, alpha=cfg.PGNN_structure.FEATURE_CHANNEL * 2, beta=0.75,
                                           k=1.0)
        self.gnn_layer = cfg.PGNN_structure.GNN_LAYER

        self.atts = nn.Parameter(torch.Tensor(np.random.rand(2, 2)))  # num_layer=2
        self.cross_iter_num = 1

        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PGNN_structure.FEATURE_CHANNEL * 2, cfg.PGNN_structure.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PGNN_structure.GNN_FEAT, cfg.PGNN_structure.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.PGNN_structure.GNN_FEAT))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.PGNN_structure.GNN_FEAT * 2, cfg.PGNN_structure.GNN_FEAT))

    # def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H, type='img'):
    def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H, type='feat'):

        # """grey image"""
        # # print("src shape", src.shape)
        # src = torch.cat((src, src, src), 1)
        # print("src shape", src.shape)               # 8*3*256*256
        # # print("tgt shape", tgt.shape)
        # tgt = torch.cat((tgt, tgt, tgt), 1)
        # # print("tgt shape", tgt.shape)



        """use feat"""
        # print("src shape", src.shape)       #8*1000*3
        src = src.squeeze(1)
        # print("src shape", src.shape)       #8*1000*3

        # print("tgt shape", tgt.shape)
        tgt = tgt.squeeze(1)
        # print("tgt shape", tgt.shape)


        if type == 'img' or type == 'image':
            # extract feature


            src_node = self.node_layers(src)      #8*512*32*32
            print("src_node", src_node.size())

            src_edge = self.edge_layers(src_node)  #8*512*16*16
            print("src_edge", src_edge.size())


            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)


            # feature normalization
            src_node = self.l2norm(src_node)            #8*512*32*32
            print("src_node", src_node.size())


            src_edge = self.l2norm(src_edge)             #8*512*16*16
            print("src_edge", src_edge.size())


            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)

            # arrange features
            U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
            F_src = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, cfg.PAIR.RESCALE)

            print("feature shape", U_src.size())     #8*512*3/4
            print("feature shape", F_src.size())     #8*512*3/4
            print("feature shape", U_tgt.size())     #8*512*3/4
            print("feature shape", F_tgt.size())     #8*512*3/4


        elif type == 'feat' or type == 'feature':
            # # print("feature shape", src.size())     #8*500*3/4
            # U_src = src[:, :src.shape[1] // 2, :]
            # # print("feature shape", U_src.size())     #8*500*3/4
            #
            # F_src = src[:, src.shape[1] // 2:, :]
            # U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            # F_tgt = tgt[:, tgt.shape[1] // 2:, :]


            U_src = src[:, :-2, :]
            print(" U_src feature shape", U_src.size())     #8*500*3/4

            F_src = src[:, -2:, :]
            print(" F_src feature shape", F_src.size())     #8*500*3/4

            U_tgt = tgt[:, :-2, :]
            F_tgt = tgt[:, -2:, :]







            # print("feature shape", U_src.size())


        else:
            raise ValueError('unknown type string {}'.format(type))

        # adjacency matrices
        A_src = torch.bmm(G_src, H_src.transpose(1, 2))
        A_tgt = torch.bmm(G_tgt, H_tgt.transpose(1, 2))
        '''
        if cfg.DATASET_FULL_NAME == "PascalVOC":
            if A_src.shape[1] == 0 or A_tgt.shape[1] == 0 or A_src.shape[1]<6 or A_tgt.shape[1]<6:
        '''
        # Motif and HyperGraph Matrices
        A_src_GHG = gen_hyper_graph2()

        A_src_hyperA_list = []
        A_src_motif2A_list = []
        A_src_motif_label_list = []
        A_src_tensor = A_src.cpu()
        for patch_len in range(A_src_tensor.shape[0]):
            A_src_Adj = A_src_tensor[patch_len]
            A_src_GHG.A = A_src_Adj
            A_src_hyperA, A_src_motif2A, A_src_motif_label = A_src_GHG.run()

            A_src_hyperA_list.append(A_src_hyperA)
            A_src_motif2A_list.append(A_src_motif2A)
            A_src_motif_label_list.append(A_src_motif_label)

        A_src_motif_label_list = torch.cat(
            [torch.from_numpy(A_src_motif_label_list[graph]) for graph in range(len(A_src_motif_label_list))],
            0)

        A_tgt_GHG = gen_hyper_graph2()
        A_tgt_hyperA_list = []
        A_tgt_motif2A_list = []
        A_tgt_motif_label_list = []
        A_tgt_tensor = A_tgt.cpu()
        for patch_len in range(A_tgt_tensor.shape[0]):
            A_tgt_Adj = A_tgt_tensor[patch_len]
            A_tgt_GHG.A = A_tgt_Adj
            A_tgt_hyperA, A_tgt_motif2A, A_tgt_motif_label = A_tgt_GHG.run()

            A_tgt_hyperA_list.append(A_tgt_hyperA)
            A_tgt_motif2A_list.append(A_tgt_motif2A)
            A_tgt_motif_label_list.append(A_tgt_motif_label)

        A_tgt_motif_label_list = torch.cat(
            [torch.from_numpy(A_tgt_motif_label_list[graph]) for graph in range(len(A_tgt_motif_label_list))],
            0)

        emb1, emb2 = torch.cat((U_src, F_src), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt), dim=1).transpose(1, 2)
        emb3, emb4 = A_src_motif_label_list, A_tgt_motif_label_list

        iteration = False
        beta = 0#0.00001
        
        if iteration:
            for i in range(self.gnn_layer-1):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2, emb3, emb4 = gnn_layer([A_src, emb1], [A_tgt, emb2], [emb3], [emb4])

                h1 = emb1

                hidden_rep3 = []
                if len(A_src_hyperA_list) != 0:

                    X3_concat = emb3

                    hidden_rep3.append(X3_concat)
                    motif2A = np.zeros((h1.shape[1] * h1.shape[0], X3_concat.shape[0]))
                    start_node_id = 0
                    start_motif_id = 0
                    elem = 0
                    for graph_idx in range(len(A_src_motif2A_list)):
                        mat = np.array(np.where(np.array(A_src_motif2A_list[graph_idx]) == 1)).T
                        elem += mat.shape[0]
                        mat[:, 0] += start_motif_id
                        mat[:, 1] += start_node_id
                        for m in mat:
                            motif2A[m[1]][m[0]] = 1
                        start_node_id += np.array(A_src_motif2A_list[graph_idx]).shape[1]
                        start_motif_id += np.array(A_src_motif2A_list[graph_idx]).shape[0]
                    motif2A = torch.FloatTensor(motif2A)

                    h3 = X3_concat.to(torch.float32)

                h33 = torch.mm(motif2A, h3).to(device)
                h33 = h33.reshape(h1.shape[0], h1.shape[1], h1.shape[2])
                #h33 = nn.AdaptiveAvgPool1d(h33.shape[2])(h33)
                
                h1 = h1 + beta * (h33)

                emb1 = h1

                h2 = emb2
                hidden_rep4 = []
                if len(A_tgt_hyperA_list) != 0:

                    X4_concat = emb4

                    hidden_rep4.append(X4_concat)
                    motif2A = np.zeros((h2.shape[1] * h2.shape[0], X4_concat.shape[0]))
                    start_node_id = 0
                    start_motif_id = 0
                    elem = 0
                    for graph_idx in range(len(A_tgt_motif2A_list)):
                        mat = np.array(np.where(np.array(A_tgt_motif2A_list[graph_idx]) == 1)).T

                        elem += mat.shape[0]
                        mat[:, 0] += start_motif_id
                        mat[:, 1] += start_node_id
                        for m in mat:
                            motif2A[m[1]][m[0]] = 1
                        start_node_id += np.array(A_tgt_motif2A_list[graph_idx]).shape[1]
                        start_motif_id += np.array(A_tgt_motif2A_list[graph_idx]).shape[0]
                    motif2A = torch.FloatTensor(motif2A)

                    h4 = X4_concat.to(torch.float32)

                h44 = torch.mm(motif2A, h4).to(device)
                
                h44 = h44.reshape(h2.shape[0], h2.shape[1], h2.shape[2])
                #h44 = nn.AdaptiveAvgPool1d(h44.shape[2])(h44)
                h2 = h2 + beta * (h44)

                emb2 = h2

            s = torch.zeros(emb1.shape[0], emb1.shape[1], emb2.shape[1], device=emb1.device)

            for it in range(self.cross_iter_num):
                i = self.gnn_layer - 2
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                emb2_new = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                emb1 = emb1_new
                emb2 = emb2_new

                i = self.gnn_layer - 1
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))

                emb1, emb2, emb3, emb4 = gnn_layer([A_src, emb1], [A_tgt, emb2], [emb3], [emb4])

                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)

                s = self.voting_layer(s, ns_src, ns_tgt)
                s = self.bi_stochastic(s, ns_src, ns_tgt)

            d, _ = self.displacement_layer(s, P_src, P_tgt)

        else:
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2, emb3, emb4 = gnn_layer([A_src, emb1], [A_tgt, emb2], [emb3], [emb4])


                h1 = emb1
                hidden_rep3 = []
                if len(A_src_hyperA_list) != 0:

                    X3_concat = emb3

                    hidden_rep3.append(X3_concat)
                    motif2A = np.zeros((h1.shape[1] * h1.shape[0], X3_concat.shape[0]))
                    start_node_id = 0
                    start_motif_id = 0
                    elem = 0
                    for graph_idx in range(len(A_src_motif2A_list)):
                        mat = np.array(np.where(np.array(A_src_motif2A_list[graph_idx]) == 1)).T

                        elem += mat.shape[0]
                        mat[:, 0] += start_motif_id
                        mat[:, 1] += start_node_id
                        for m in mat:
                            motif2A[m[1]][m[0]] = 1
                        start_node_id += np.array(A_src_motif2A_list[graph_idx]).shape[1]
                        start_motif_id += np.array(A_src_motif2A_list[graph_idx]).shape[0]
                    motif2A = torch.FloatTensor(motif2A)

                    h3 = X3_concat.to(torch.float32)

                h33 = torch.mm(motif2A, h3).to(device)
                h33 = h33.reshape(h1.shape[0], h1.shape[1], h1.shape[2])
                h33 = nn.AdaptiveMaxPool1d(h33.shape[2])(h33)
                '''
                for i in range(len(h33)):
                     print(torch.max(h33[i]), torch.min(h33[i]))
                     '''
                h1 = h1 + beta * h33
                
                emb1 = h1

                h2 = emb2

                hidden_rep4 = []
                if len(A_tgt_hyperA_list) != 0:

                    X4_concat = emb4

                    hidden_rep4.append(X4_concat)
                    motif2A = np.zeros((h2.shape[1] * h2.shape[0], X4_concat.shape[0]))
                    start_node_id = 0
                    start_motif_id = 0
                    elem = 0
                    for graph_idx in range(len(A_tgt_motif2A_list)):
                        mat = np.array(np.where(np.array(A_tgt_motif2A_list[graph_idx]) == 1)).T

                        elem += mat.shape[0]
                        mat[:, 0] += start_motif_id
                        mat[:, 1] += start_node_id
                        for m in mat:
                            motif2A[m[1]][m[0]] = 1
                        start_node_id += np.array(A_tgt_motif2A_list[graph_idx]).shape[1]
                        start_motif_id += np.array(A_tgt_motif2A_list[graph_idx]).shape[0]
                    motif2A = torch.FloatTensor(motif2A)

                    h4 = X4_concat.to(torch.float32)

                h44 = torch.mm(motif2A, h4).to(device)
                h44 = h44.reshape(h2.shape[0], h2.shape[1], h2.shape[2])
                
                                
                h44 = h44.reshape(h2.shape[0], h2.shape[1], h2.shape[2])
                h44 = nn.AdaptiveMaxPool1d(h44.shape[2])(h44)
                '''
                for i in range(len(h44)):
                    print(torch.max(h44[i]), torch.min(h44[i]))
                resume2 =False
                assert resume2,"break"
                '''
                h2 = h2 + beta * h44

                emb2 = h2

                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)
                s = self.voting_layer(s, ns_src, ns_tgt)
                s = self.bi_stochastic(s, ns_src, ns_tgt)
                
                if i == self.gnn_layer - 2:
                    cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                    emb1_new = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                    emb2_new = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                    emb1 = emb1_new
                    emb2 = emb2_new
                

            d, _ = self.displacement_layer(s, P_src, P_tgt)

        return s, d
