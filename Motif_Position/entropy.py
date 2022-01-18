# encoding: utf-8
import numpy as np
from sympy import *
import math
import re
Nm = 8


def get_Amount_of_Motif():
    array = open('./data2/CountMotif.csv').readlines()
    matrix = []
    for line in array:
        line = line.strip('\r\n').split(',')
        line = [int(x) for x in line]
        matrix.append(line)
    matrix = np.array(matrix)
    return matrix


def read_data(i, filename):
    array = open(filename).readlines()
    #print(i)
    N = int(array[i])
    return N


def devide(N):
    n = [[0] * N for i in range(0, N)]
    for i in range(N):
        for j in range((i + 1) / 2):
            if i == 0:
                n[i][j] = 1
            else:
                if j == 0:
                    n[i][j] = sum(n[i - 1])
                else:
                    if i + 1 == 2 * (j + 1):
                        n[i][j] = n[i - j - 1][j]
                    else:
                        n[i][j] = n[i - j - 1][j] + 1
        n[i][i] = 1
    return n


def calEntropy(n, N, dN):
    l = [2, 3, 3, 4, 4, 4, 5, 5]
    e = [1, 2, 3, 3, 3, 4, 4, 4]
    iso = [1, 3, 1, 12, 4, 1, 20, 5]
    K = 1.0 / 100000
    T = 100.0
    DELTA = 256

    beta = symbols('BETA')
    pi = 3.1415926
    Integral = sqrt(beta) * sqrt(pi) * DELTA
    Entropy = []
    r = 27.0
    global Nm
    for i in range(Nm):
        m_integral = iso[i] * (Integral ** e[i]) * r ** (l[i] - e[i])
        if N - l[i] * n[i] > 0:
            logZ = n[i] * (log(m_integral) - log(n[i]) - l[i] * log(l[i]) + 1) + (N - l[i] * n[i]) * (
                    log(r) - log(N - l[i] * n[i]) + 1)
        else:
            logZ = n[i] * (-math.log(n[i]) + 1 + log(m_integral) - math.log(factorial(l[i])))
        logZ += math.log(dN)
        U = diff(logZ, beta)
        E = logZ + U * beta
        E = E.subs(beta, 1.0 / (K * T))
        Entropy.append(float(E))
    return Entropy


def graphEntropy(motifNumber, nodN):
    dN = read_data(nodN - 1, 'Motif_Position/devide_10000_Nodes.csv')
    return calEntropy(motifNumber, nodN, dN)


def node_entropy(motif_number, graph_entropy, node_occupation):
    node_entropy_vactor = [0 for i in range(len(node_occupation))]
    for index, node in enumerate(node_occupation):
        for item in node:
            node_entropy_vactor[index] += graph_entropy[int(item)] / motif_number[int(item)]

    return node_entropy_vactor


def graphlet_entropy(n):
    l = [2, 3, 3, 3, 4, 4, 4, 4]
    e = [1, 2, 2, 3, 3, 3, 3, 3]
    K = 1.0 / 100000
    T = 100.0

    S = 1000

    DELTA = 256

    beta = symbols('BETA')
    pi = 3.1415926
    Integral = sqrt(beta) * sqrt(pi) * DELTA
    Entropy = []
    r = 10
    Nm = len(n)
    for i in range(Nm):
        if n[i] == 0:
            Entropy.append(float(0))
            continue
        m_integral = (Integral ** e[i % 8]) * r ** (l[i % 8] - e[i % 8])
        logZ = n[i] * (-math.log(n[i]) + 1 + log(m_integral) - math.log(factorial(l[i % 8])))
        U = diff(logZ, beta)
        E = logZ + U * beta
        E = E.subs(beta, S)
        Entropy.append(float(E))
    return Entropy
