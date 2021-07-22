import torch
import torch.nn as nn
import torch.utils.data as data
from torchsummary import summary
from dataset.ModelNet40 import ModelNet40
from ..model.TNet import input_transform_feature, feature_transform
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PointNet(nn.Module):
    def __init__(self, classes=40, Tnet=True):
        super(PointNet, self).__init__()
        self.classes = classes
        self.input_transform_feature = input_transform_feature()
        self.feature_transform = feature_transform()
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
        )

        self.classifer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, self.classes),

        )
        self.Tnet=Tnet

    def forward(self, x):
        batchsize = x.size()[0]  # x:Bx3xN
        sample_points = x.size()[2]
        '''
        第一个TNet
        '''
        input_transform_feature = self.input_transform_feature(x)  # torch.Size([Batchsize, 3, 3])

        x = x.permute(0, 2, 1)  # BxNx3
        x = torch.bmm(x, input_transform_feature)  # Bx[(Nx3)x(3x3)]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)  # torch.Size([B, 64, 1024])
        '''
        第二个TNet
        '''
        if self.Tnet:
            feature_transform = self.feature_transform(x) # Bx64x64
            x = x.permute(0, 2, 1) # Bx1024x64
            x=torch.bmm(x, feature_transform)   # Bx[(Nx64)x(64x64)]
            x = x.permute(0, 2, 1) #torch.Size([B, 64, N=1024])
        else:
            feature_transform=None

        x = self.features(x)  # Bx1024x(N=1024)
        x = torch.max(x, 2, keepdim=True)[0]  # Bx1024x1
        x = x.squeeze()
        x = self.classifer(x)  # torch.Size([B, 40])
        return F.log_softmax(x, dim=1)

def main():
    sim_data = torch.rand(5,3,2048).to(device)
    root = r'/home/sirb/Documents/ModelNet40_ply'
    train_data = ModelNet40(root,sample_points=1024,split='train')
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=10,shuffle=True,num_workers=4)
    Net = PointNet().to(device)
    summary(Net, (3, 2048))

    for index, (x, y) in enumerate(train_loader):
        x, y = x.permute(0, 2, 1).to(device), y.to(device)  # x数据是BxNX3 变为 Bx3xN
        out = Net(x)
# if __name__=="__main__":
#    main()
