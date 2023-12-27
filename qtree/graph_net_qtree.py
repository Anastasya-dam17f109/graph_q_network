import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        return (x1,x2,x3,x4)


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

        for i in range(depth):
            layers.append(np.zeros((buf_width,buf_width), dtype=int))
            for k in range(buf_width*buf_width):
                layers[i][k // buf_width, k % buf_width] = k + global_idx
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
        print(layers)
        print(idxes)
gen_graph_undirect(4, 2)

#net = Net()



'''
https://github.com/gordicaleksa/pytorch-GAT/blob/main/models/definitions/GAT.py
https://github.com/gillesvntnu/GCN_multistructure/blob/main/nets/CNN_GCN.py
https://telegra.ph/Grafovye-svertochnye-seti-vvedenie-v-GNN-10-16
https://github.com/rampasek/GraphGPS

'''