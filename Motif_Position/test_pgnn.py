import copy
import re
import numpy as np
import torch
import torch.nn.functional as F
from Motif_Position.utils_pgnn import *
from Motif_Position.MotifCount import *
from Motif_Position.entropy import graphEntropy


def concateTensor(Graph_ex):
    tempa, tempb,tempc = caculate(Graph_ex)

    tempA = torch.zeros(len(tempa), tempa[0].shape[0], tempa[0].shape[1])
    # print(Graph_ex.shape[0])
    for i in range(len(tempa)):
        tempA[i] = tempa[i]
    # print(tempA)
    tempB = torch.zeros(len(tempa), tempa[0].shape[0], tempa[0].shape[1]+tempa[0].shape[0])
    for i in range(len(tempa)):
        tempB[i] = torch.cat((Graph_ex[i].cpu(), tempA[i]), -1)
    # print(tempB)
    
    count_sum = []
    for i in range(len(tempa)):
        tempA[i] = tempa[i]
        count_sum.append(np.sum(tempA[i].numpy()))
    count_count_sum = np.sum(count_sum)
    
    tempC = torch.zeros(Graph_ex.shape)
    for i in range(len(tempa)):
        if count_count_sum!=0:
          att = count_sum[i]/count_count_sum
          tempC[i] = tempc[i]+Graph_ex[i].cpu()
        else:
          tempC[i] = Graph_ex[i].cpu()
    return tempC.to(device)


def concateTensor2(Graph_ex):
    tempa, tempb,tempc = caculate(Graph_ex)

    tempA = torch.zeros(len(tempa), tempa[0].shape[0], tempa[0].shape[1])
    # print(Graph_ex.shape[0])
    count_sum = []
    for i in range(len(tempa)):
        tempA[i] = tempa[i]
        count_sum.append(np.sum(tempA[i].numpy()))
    # print(tempA)
    #print(count_sum)
    count_count_sum = np.sum(count_sum)
    tempB = torch.zeros(len(tempa), tempa[0].shape[0], tempa[0].shape[1]+tempa[0].shape[0])
    for i in range(len(tempa)):
        tempB[i] = torch.cat((Graph_ex[i].cpu(), tempA[i]), -1)

    # print(tempA)
    # print(tempB)
    tempC = torch.zeros(Graph_ex.shape)
    for i in range(len(tempa)):
        #att = count_sum/count_count_sum
        # print(att)
        tempC[i] = Graph_ex[i].cpu()*tempc[i]
    return tempC.to(device)
    
def concateTensor3(Graph_ex):
    tempa, tempb, tempc = caculate(Graph_ex)

    tempC = torch.zeros(Graph_ex.shape)
    for i in range(len(tempa)):
        # print(att)
        tempC[i] = tempc[i]
    # print(tempC)
    return tempC.to(device)


def count_entropy(Graph_ex):
    length = len(Graph_ex)

    Adj = convert(Graph_ex, length)
    store = []
    Nm = 8
    for i in range(length):
        store.append(np.array(nx.adjacency_matrix(Adj[i]).todense()))
    # return an adj matrix for each graph
    count_node_entopy = []
    # convert string to int
    for i in range(length):
        num_motif, num_node = count_Motifs(store[i])
        # print(num_motif,"\t",num_node)
        temp = []
        # print(num_node)
        for i in range(len(num_node)):
            if num_node[i] != '':
                length2 = len(num_node[i])
                value = int(num_node[i])

                result = []
                while value:
                    result.append(value % 10)
                    value = value // 10

                result.reverse()
                temp.append(result)
        # print(temp)

        # count 1 to each node
        a = np.zeros((len(num_node), Nm), float)
        for i, temp1 in enumerate(temp):
            for k in range(len(temp1)):
                a[i][temp1[k] - 1] = 1
        # print(a)

        # cout node appear in each motif
        count_number_motif = []
        count_number_motif1 = 0
        for j in range(Nm):
            for i in range(len(num_node)):
                count_number_motif1 += a[i][j]
            count_number_motif.append(count_number_motif1)
            count_number_motif1 = 0
        # print(count_number_motif)

        motif_entropy = graphEntropy(num_motif, len(num_node))
        # print(motif_entropy)
        entropy_count = []
        for i in range(Nm):
            if count_number_motif[i] != 0:
                entropy_count.append(motif_entropy[i] / count_number_motif[i])
            else:
                entropy_count.append(0)
        # print(entropy_count)

        # motifµÄentropy
        for i in range(Nm):
            for j in range(len(num_node)):
                if a[j][i] == 1:
                    a[j][i] = entropy_count[i]
        # print(a)

        # deliver to 1
        count1 = 0
        count_every_vector = []
        for i in range(len(num_node)):
            for j in range(Nm):
                count1 += a[i][j]
            count_every_vector.append(count1)
            count1 = 0
        # print(count_every_vector)
        count_node_entopy.append(count_every_vector)
    count2 = [0] * len(count_node_entopy[0])

    for m in range(len(count_node_entopy[0])):
        for k in range(len(count_node_entopy)):
            count2[m] += count_node_entopy[k][m]
    for mk in range(len(count2)):
        if np.sum(count2) != 0:
            count2[mk] /= np.sum(count2)
    # print(count2)
    temp_tensor = torch.zeros(length, Graph_ex[0].shape[0], Graph_ex[0].shape[1])

    for i in range(length):
        temp_tensor[i] = torch.from_numpy(np.identity(Graph_ex[0].shape[0]))
    # print(length)
    if np.sum(count2) != 0:
        for i in range(length):
            diag = np.diag(np.array(count2))
            # print(diag)
            temp_tensor[i] = torch.from_numpy(diag)
    else:
        for i in range(length):
            diag = np.identity(Graph_ex[0].shape[0])
            # print(diag)
            temp_tensor[i] = torch.from_numpy(diag)

    return temp_tensor.to(device)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def concate(Graph_ex):
    length = len(Graph_ex)
    count_node_en = count_entropy(Graph_ex).cpu()
    count_dist_graph = concateTensor3(Graph_ex).cpu()
    count_att_graph = concateTensor2(Graph_ex).cpu()
    position = sub_dist(Graph_ex).cpu()
    concate_four = torch.zeros(length, 2 * Graph_ex[0].shape[0], 2 * Graph_ex[0].shape[1])
    matrix = torch.zeros(length, Graph_ex[0].shape[0], Graph_ex[0].shape[1])
    # print(count_node_en.shape, count_att_graph.shape, count_dist_graph.shape, concate_four.shape)
    for i in range(length):
        if np.all(Graph_ex[i].cpu().numpy() == 0):
            Graph_ex[i] = torch.from_numpy(np.identity(Graph_ex[0].shape[0])).to(device)
        temp_concat1 = torch.cat((Graph_ex[i].cpu(), Graph_ex[i].cpu()), -1)
        temp_concat2 = torch.cat((count_dist_graph[i], count_node_en[i]), -1)

        temp_3 = np.array(Graph_ex[i].cpu())
        temp_4 = np.array(count_node_en[i])
        temp_5 = np.array(count_att_graph[i])
        temp_6 = np.array(position[i])
        #matrix[i] = torch.from_numpy(temp_3*temp_4)
        matrix[i] = torch.from_numpy(temp_4)
    return matrix.to(device)

def getDist(Graph_ex):
    tempa, tempb, tempc = caculate(Graph_ex)
    zeroA = torch.zeros(len(tempa), tempa[0].shape[0], tempa[0].shape[1])
    zeroB = torch.zeros(len(tempb), tempb[0].shape[0], tempb[0].shape[1])
    for i in range(len(tempa)):
        zeroA[i] = tempa[i].cpu()
        zeroB[i] = tempb[i].cpu()
    return zeroA.to(device), zeroB.to(device)
    
    
def sub_dist(Graph_ex):
    a, b = getDist(Graph_ex)
    # print(a[0].unsqueeze(-1))
    sub_gh = torch.zeros(len(b),b[0].shape[0],b[0].shape[1])
    sub_gh2 = torch.zeros(len(a),a[0].shape[1],a[0].shape[0])
    sub_gh3 = torch.zeros(len(b),Graph_ex[0].shape[0],Graph_ex[0].shape[1])
    for i in range(len(Graph_ex)):
        for j in range(Graph_ex[0].shape[0]):
            for k in range(a[0].shape[1]):
                sub_gh[i][j][k] = Graph_ex[i][j][int(b[i][j][k])]
    for i in range(len(Graph_ex)):
        sub_gh2[i] = torch.from_numpy(a[i].cpu().numpy().reshape(b[0].shape[1],b[0].shape[0]))

    for i in range(len(Graph_ex)):
        sub_gh3[i] = torch.from_numpy(np.dot(sub_gh[i].numpy(),sub_gh2[i].numpy()))
    #print(sub_gh3.cuda())
    return sub_gh3.to(device)

