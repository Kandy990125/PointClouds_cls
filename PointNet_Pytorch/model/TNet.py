import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
功能：第一个T-Net,相当于将输入的数据旋转.为每个BatchSize都得到一个3x3的矩阵
输入：Bx3xN
输出：Bx3x3
'''
class input_transform_feature(nn.Module):
    def __init__(self):
        super(input_transform_feature, self).__init__()

        self.Conv1=nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024),
        )

        self.FC1=nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256,9)
        )

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.Conv1(x)
        x = torch.max(x, 2, keepdim=True)[0] # torch.Size([Batch_size, 1024, 1])
        x = x.view(-1, 1024)
        x1 = self.FC1(x)  # torch.Size([Batch_size, 9])
        bias = autograd.Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1,9).repeat(batchsize, 1)
        if x.is_cuda:
            bias = bias.cuda()
        x1 = x1+bias
        x1 = x1.view(-1, 3, 3)  # torch.Size([Batch_size, 3, 3])
        return x1


'''
功能：第二个T-Net,为每个BatchSize都得到一个64x64的矩阵
输入：Bx64xN
输出：Bx64x64
'''
class feature_transform(nn.Module):
    def  __init__(self):
        super(feature_transform,self).__init__()

        self.Conv1=nn.Sequential(
            nn.Conv1d(64,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024),
        )

        self.FC1=nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256,64*64)
        )

    def forward(self,x):
        batchsize=x.size()[0]
        x=self.Conv1(x)
        x=torch.max(x,2,keepdim=True)[0]
        x=x.view(-1,1024)
        x1=self.FC1(x) # torch.Size([Batch_size, 64*64])
        bias = autograd.Variable(torch.from_numpy(np.eye(64).flatten().astype(np.float32))).view(1, 64 * 64).repeat(batchsize, 1)
        if x.is_cuda:
            bias=bias.cuda()
        x1=x1+bias
        x1=x1.view(-1,64,64)  # torch.Size([Batch_size, 64, 64])
        return x1

def main():
    data = torch.rand(10, 64, 1024).to(device)
    STN64 = feature_transform().to(device)
    STN64(data)


# if __name__ == "__main__":
#     main()
