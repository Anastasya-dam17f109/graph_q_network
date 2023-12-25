import torch.nn as nn
import torch.nn.functional as F

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


net = Net()

        


'''
https://github.com/gordicaleksa/pytorch-GAT/blob/main/models/definitions/GAT.py
https://github.com/gillesvntnu/GCN_multistructure/blob/main/nets/CNN_GCN.py
https://telegra.ph/Grafovye-svertochnye-seti-vvedenie-v-GNN-10-16
https://github.com/rampasek/GraphGPS

'''