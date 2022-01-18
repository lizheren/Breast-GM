import numpy as np
import csv
import copy

# number of motif

Nm = 8


def clean_node(a, index):
    a[index].fill(0)
    a[:, index].fill(0)


def count_star(A, N, neiN, node_occupation, motif):
    n = 0
    a = copy.copy(A)
    for i in range(N):
        next_list = np.nonzero(a[i])[0]

        if (len(next_list) >= neiN):
            n += 1
            node_occupation[i] += str(motif)
            clean_node(a, i)
            for j in range(neiN):
                node_occupation[next_list[j]] += str(motif)
                clean_node(a, next_list[j])
    return n


def find_next(a, i, rest, stack):
    if rest == 0:
        return 1

    next_index_list = np.nonzero(a[i])[0]

    for next_index in next_index_list:
        if next_index not in stack:
            stack.append(next_index)
            if find_next(a, next_index, rest - 1, stack) > 0:
                return 1

    stack.pop()
    return -1


def count_chain(A, N, len, node_occupation, motif):
    n = 0
    a = copy.copy(A)
    for i in range(N):
        stack = [i]
        if find_next(a, i, len - 1, stack) > 0:
            # print('find chain ', stack)
            for j in stack:
                node_occupation[j] += str(motif)
                clean_node(a, j)
            n += 1
    return n


def count_triangle(A, N, node_occupation):
    n = 0
    a = copy.copy(A)
    for i in range(N):
        for j in range(i, N):
            if a[i][j] > 0:
                for k in range(j, N):
                    if a[j][k] > 0 and a[k][i] > 0:
                        n += 1
                        node_occupation[i] += str(3)
                        node_occupation[j] += str(3)
                        node_occupation[k] += str(3)
                        clean_node(a, i)
                        clean_node(a, j)
                        clean_node(a, k)
    return n


def count_quadrangle(A, N, node_occupation):
    n = 0
    a = copy.copy(A)
    for i in range(N):
        stack = [i]
        for j in np.nonzero(a[i])[0]:
            stack.append(j)
            for k in np.nonzero(a[j])[0]:
                if k not in stack:
                    stack.append(k)
                    for l in np.nonzero(a[k])[0]:
                        if l not in stack and a[l][i] > 0:
                            n += 1
                            node_occupation[i] += str(6)
                            node_occupation[j] += str(6)
                            node_occupation[k] += str(6)
                            node_occupation[l] += str(6)
                            clean_node(a, i)
                            clean_node(a, j)
                            clean_node(a, k)
                            clean_node(a, l)
                    stack.pop()
            stack.pop()
    return n


def count_Motifs(A):
    nodN = len(A)
    # print(nodN)
    node_occupation = ['' for i in range(nodN)]
    rd = np.argsort(sum(np.transpose(A)))
    # node按照degree升序排列
    rdA = A[rd]
    # print(rdA)
    rdA[:, ] = rdA[:, rd]
    # print(rdA)

    Nm_1 = count_chain(rdA, nodN, 2, node_occupation, 1)
    Nm_2 = count_chain(rdA, nodN, 3, node_occupation, 2)
    Nm_3 = count_triangle(rdA, nodN, node_occupation)
    Nm_4 = count_chain(rdA, nodN, 4, node_occupation, 4)
    Nm_5 = count_star(rdA, nodN, 3, node_occupation, 5)
    Nm_6 = count_quadrangle(rdA, nodN, node_occupation)
    Nm_7 = count_chain(rdA, nodN, 5, node_occupation, 7)
    Nm_8 = count_star(rdA, nodN, 4, node_occupation, 8)
    num = [Nm_1, Nm_2, Nm_3, Nm_4, Nm_5, Nm_6, Nm_7, Nm_8]

    # 将node顺序复原
    node_occupation_new = ['' for i in range(nodN)]
    for index in range(nodN):
        index_new = np.where(rd == index)[0][0]
        # 返回condition索引
        node_occupation_new[index] = node_occupation[index_new]
    # print(num, node_occupation_new)
    return num, node_occupation_new
