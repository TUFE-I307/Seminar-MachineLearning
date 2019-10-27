import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import scipy.linalg as linalg


def partition(G, k, normalized=False):
    A = nx.to_numpy_array(G)
    D = degree_matrix(G)
    L = D - A
    Dn = np.power(np.linalg.matrix_power(D, -1), 0.5)
    L = np.dot(np.dot(Dn, L), Dn)
    if normalized:
        pass
    eigvals, eigvecs = linalg.eig(L)
    n = len(eigvals)

    dict_eigvals = dict(zip(eigvals, range(0, n)))
    k_eigvals = np.sort(eigvals)[0:k]
    eigval_indexs = [dict_eigvals[k] for k in k_eigvals]
    k_eigvecs = eigvecs[:, eigval_indexs]
    result = KMeans(n_clusters=k).fit_predict(k_eigvecs)
    return result


def degree_matrix(G):
    n = G.number_of_nodes()
    V = [node for node in G.nodes()]
    D = np.zeros((n, n))
    for i in range(n):
        node = V[i]
        d_node = G.degree(node)
        D[i][i] = d_node
    return np.array(D)


if __name__ == '__main__':
    G = nx.Graph()
    node_list = [i for i in range(1, 17)]
    edge_list = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6), (4, 5),
                 (7, 8), (7, 9), (7, 10), (7, 11), (8, 9), (8, 10), (9, 10),
                 (12, 13), (12, 14), (12, 15), (12, 16), (13, 14), (13, 15), (13, 16), (14, 15), (14, 16), (15, 16),
                 (5, 7), (1, 8), (1, 12), (2, 13), (7, 13), (8, 12)]
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    k = 3
    res = partition(G, k)
