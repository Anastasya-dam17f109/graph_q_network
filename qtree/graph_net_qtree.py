import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch

# encoder

class Net(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(3, num_feat, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, )

    def forward(self, x):
        x1 = self.pool1(F.relu(self.conv1(x)))
        x2 = self.pool2(F.relu(self.conv2(x1)))
        x3 = self.pool3(F.relu(self.conv3(x2)))
        x4 = F.relu(self.conv4(x3))
        x1 = torch.reshape(x1, (x1.size[1]*x1.size[2],x1.size[3]))
        x1 = torch.split(x1, x1.size[1])
        x2 = torch.reshape(x2, (x2.size[1] * x2.size[2], x2.size[3]))
        x2 = torch.split(x2, x2.size[1])
        x3 = torch.reshape(x3, (x3.size[1] * x3.size[2], x3.size[3]))
        x3 = torch.split(x3, x3.size[1])
        x4 = torch.reshape(x4, (x4.size[1] * x4.size[2], x4.size[3]))
        x4 = torch.split(x4, x4.size[1])
        return x1,x2,x3,x4


def gen_graph_direct(width, depth):
    print(width % 2**depth)
    if width % 2**depth==0:
        buf_width=width // 2**depth
        layers = []
        idxes=[]
        global_idx=0
        for i in range(depth):
            layers.append(np.zeros((buf_width,buf_width), dtype=int))
            for k in range(buf_width*buf_width):
                layers[i][k//buf_width, k%buf_width]=k+global_idx
            for j in range(buf_width):
                for k in range(buf_width):
                    if i != 0:
                        idxes.append((layers[i-1][j // 2, k // 2],layers[i][j,k]))
                    if     buf_width > 1:
                        if k==0 or k == buf_width-1:
                            if k==0:
                                if j ==0 or j == buf_width-1:
                                    if j ==0:
                                        idxes.append((layers[i][j, k], layers[i][j, k + 1]))
                                        idxes.append((layers[i][j, k], layers[i][j+1, k + 1]))
                                        idxes.append((layers[i][j, k], layers[i][j + 1, k ]))
                                    else:
                                        idxes.append((layers[i][j, k], layers[i][j, k + 1]))
                                        idxes.append((layers[i][j, k], layers[i][j -1, k + 1]))
                                        idxes.append((layers[i][j, k], layers[i][j - 1, k]))
                                else:
                                    idxes.append((layers[i][j,k],layers[i][j,k + 1]))
                                    idxes.append((layers[i][j, k], layers[i][j + 1, k ]))
                                    idxes.append((layers[i][j, k], layers[i][j - 1, k]))
                                    idxes.append((layers[i][j, k], layers[i][j + 1, k + 1]))
                                    idxes.append((layers[i][j, k], layers[i][j - 1, k + 1]))
                            else:
                                if j ==0 or j == buf_width-1:
                                    if j ==0:
                                        idxes.append((layers[i][j, k], layers[i][j, k - 1]))
                                        idxes.append((layers[i][j, k], layers[i][j+1, k - 1]))
                                        idxes.append((layers[i][j, k], layers[i][j + 1, k ]))
                                    else:
                                        idxes.append((layers[i][j, k], layers[i][j, k - 1]))
                                        idxes.append((layers[i][j, k], layers[i][j -1, k - 1]))
                                        idxes.append((layers[i][j, k], layers[i][j - 1, k]))
                                else:
                                    idxes.append((layers[i][j,k],layers[i][j,k - 1]))
                                    idxes.append((layers[i][j, k], layers[i][j + 1, k ]))
                                    idxes.append((layers[i][j, k], layers[i][j - 1, k]))
                                    idxes.append((layers[i][j, k], layers[i][j + 1, k - 1]))
                                    idxes.append((layers[i][j, k], layers[i][j - 1, k - 1]))
                        elif j == 0 or j == buf_width - 1:
                            if j == 0:
                                idxes.append((layers[i][j, k], layers[i][j, k - 1]))
                                idxes.append((layers[i][j, k], layers[i][j + 1, k - 1]))
                                idxes.append((layers[i][j, k], layers[i][j + 1, k]))
                                idxes.append((layers[i][j, k], layers[i][j, k + 1]))
                                idxes.append((layers[i][j, k], layers[i][j + 1, k + 1]))
                            else:
                                idxes.append((layers[i][j, k], layers[i][j, k - 1]))
                                idxes.append((layers[i][j, k], layers[i][j - 1, k - 1]))
                                idxes.append((layers[i][j, k], layers[i][j - 1, k]))
                                idxes.append((layers[i][j, k], layers[i][j, k + 1]))
                                idxes.append((layers[i][j, k], layers[i][j - 1, k + 1]))
                        else:
                            idxes.append((layers[i][j, k], layers[i][j, k - 1]))
                            idxes.append((layers[i][j, k], layers[i][j + 1, k - 1]))
                            idxes.append((layers[i][j, k], layers[i][j + 1, k]))
                            idxes.append((layers[i][j, k], layers[i][j, k + 1]))
                            idxes.append((layers[i][j, k], layers[i][j + 1, k + 1]))
                            idxes.append((layers[i][j, k], layers[i][j - 1, k - 1]))
                            idxes.append((layers[i][j, k], layers[i][j - 1, k]))
                            idxes.append((layers[i][j, k], layers[i][j - 1, k + 1]))
            global_idx += buf_width*buf_width
            buf_width *= 2
        print(layers)
        print(idxes)

def gen_graph_undirect(width, depth):
    print(width % 2**depth)
    if width % 2**depth == 0:
        buf_width = width // 2**depth
        layers = []
        idxes = set()
        global_idx=0
        def add_new_idx(l, i,j,k):
            for iter in range(len(l)):
                idx_j1, idx_k1, idx_j2, idx_k2 = j, k, l[iter][0], l[iter][1]
                if layers[i][idx_j1, idx_k1] > layers[i][idx_j2, idx_k2]:
                    idx_j1, idx_k1, idx_j2, idx_k2 = idx_j2, idx_k2, idx_j1, idx_k1
                idxes.add((layers[i][idx_j1, idx_k1], layers[i][idx_j2, idx_k2]))

        for i in range(depth+1):
            layers.append(np.zeros((buf_width,buf_width), dtype=int))
            for k in range(buf_width*buf_width):
                layers[i][k // buf_width, k % buf_width] = k + global_idx-1
            for j in range(buf_width):
                for k in range(buf_width):
                    if i != 0:
                        idx_j1, idx_k1, idx_j2, idx_k2 = j // 2, k // 2, j, k
                        if layers[i-1][idx_j1, idx_k1] > layers[i][idx_j2, idx_k2]:
                            idx_j1, idx_k1, idx_j2, idx_k2 = idx_j2, idx_k2, idx_j1, idx_k1

                        idxes.add((layers[i-1][idx_j1, idx_k1], layers[i][idx_j2, idx_k2]))
                    if buf_width > 1:
                        if k==0 or k == buf_width-1:
                            if k==0:
                                if j ==0 or j == buf_width-1:
                                    if j ==0:
                                        l = [[j, k + 1], [j+1, k + 1], [j + 1, k]]
                                        add_new_idx(l, i,j,k)
                                    else:
                                        l = [[j, k + 1], [j - 1, k + 1], [j - 1, k]]
                                        add_new_idx(l, i, j, k)

                                else:
                                    l = [[j,k + 1], [j + 1, k], [j - 1, k], [j + 1, k + 1], [j - 1, k + 1]]
                                    add_new_idx(l, i, j, k)
                            else:
                                if j ==0 or j == buf_width-1:
                                    if j ==0:
                                        l = [[j, k - 1], [j + 1, k - 1], [j + 1, k]]
                                        add_new_idx(l, i, j, k)
                                    else:
                                        l = [[j, k - 1], [j - 1, k - 1], [j - 1, k]]
                                        add_new_idx(l, i, j, k)
                                else:
                                    l = [[j, k - 1], [j + 1, k], [j - 1, k], [j + 1, k - 1], [j - 1, k - 1]]
                                    add_new_idx(l, i, j, k)
                        elif j == 0 or j == buf_width - 1:
                            if j == 0:
                                l = [[j, k - 1], [j + 1, k - 1], [j + 1, k], [j, k + 1], [j + 1, k + 1]]
                                add_new_idx(l, i, j, k)
                            else:
                                l = [[j, k - 1], [j - 1, k - 1], [j - 1, k], [j, k + 1], [j - 1, k + 1]]
                                add_new_idx(l, i, j, k)
                        else:
                            l = [[j, k - 1], [j + 1, k - 1], [j + 1, k], [j, k + 1], [j + 1, k + 1], [j - 1, k - 1], [j - 1, k], [j - 1, k + 1]]
                            add_new_idx(l, i, j, k)
            global_idx += buf_width*buf_width
            buf_width *= 2

        layers.pop(0)

        idxes2 = set()
        for i in idxes:

            if i[0] != -1 and i[1] != -1:

                idxes2.add(i)
        return layers, list(idxes2)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def create_adj(width, height):
    nodes, edges = gen_graph_undirect(width, height)
    edges = np.array(edges)
    matr_shape = max([np.max(i) for i in nodes]) + 1

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:, 1])),
                         shape=(matr_shape,matr_shape), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_propg = np.linalg.inv(sp.eye(adj.shape[0]).todense() - 0.5 * normalize_adj(adj).todense())
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    print(adj)

    adj = torch.FloatTensor(np.array(adj.todense()))
    adj_propg = torch.FloatTensor(np.array(adj_propg))
    return adj, adj_propg

#create_adj(8,3)


'''
https://github.com/gordicaleksa/pytorch-GAT/blob/main/models/definitions/GAT.py
https://github.com/gillesvntnu/GCN_multistructure/blob/main/nets/CNN_GCN.py
https://telegra.ph/Grafovye-svertochnye-seti-vvedenie-v-GNN-10-16
https://github.com/rampasek/GraphGPS

'''