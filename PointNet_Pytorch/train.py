import torch
import torch.nn as nn
import numpy as np
import random
import os
from PointNet_Pytorch.model.pointnet import PointNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
blue = lambda x: '\033[94m' + x + '\033[0m'


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def main():
    seed_torch(1234)
    batch_size = 64
    # root = r'/home/sirb/Documents/ModelNet40_ply'
    # train_data = ModelNet40(root=root, sample_points=2048, split='train')
    # test_data = ModelNet40(root=root, sample_points=2048, split='test')
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    from dataset.ModelNet40 import get_data_loader
    train_loader = get_data_loader(train=True, batch_size=batch_size)
    test_loader = get_data_loader(train=False, batch_size=batch_size)

    # print('训练数据{}\t测试数据{}'.format(len(train_data), len(test_data)))
    # print(train_data.classes)
    # print('总共有:', len(train_data.classes), '类')

    net = PointNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)  # 对学习率进行调整
    loss_function = nn.NLLLoss()
    epochs = 401
    # num_batch = len(train_data)/batch_size
    res = []
    '''
    训练开始
    '''
    for epoch in range(epochs):
        scheduler.step()
        # for idx, (x, label) in enumerate(train_loader):
        #     x, label = x.permute(0, 2, 1).to(device), label[:, 0].to(device)  # x数据是BxNX3 变为 Bx3xN
        for x, label in train_loader:
            x = x.permute(0, 2, 1).to(device)
            # print(x)
            label = label.to(device)
            out = net(x)
            loss = loss_function(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_choice = out.data.max(1)[1]
            correct = pred_choice.eq(label.data).cpu().sum()
            # print('[%d: %d/%d] train loss: %f accuracy: %f' %
            # (epoch, idx, num_batch, loss.item(), correct.item() / float(batch_size)))
            print('[%d] train loss: %f accuracy: %f' % (epoch, loss.item(), correct.item() / float(batch_size)))

        if epoch % 5 == 0:
            total_correct = 0
            total_testset = 0
            # for idx, (x, label) in enumerate(test_loader):
            #     x, label = x.permute(0, 2, 1).to(device), label[:, 0].to(device)  # x数据是BxNX3 变为 Bx3xN
            for x, label in train_loader:
                x = x.permute(0, 2, 1).to(device)
                label = label.to(device)
                out = net(x)
                pred_choice = out.data.max(1)[1]
                correct = pred_choice.eq(label.data).cpu().sum()
                total_correct += correct.item()
                total_testset += x.size()[0]
            print("The {} is {}.\t Test accuracy {}".format(blue('epoch'), epoch, total_correct / float(total_testset)))
            res.append(total_correct / float(total_testset))
            torch.save(net.state_dict, 'out/model_epoch%d.pth' % (epoch))
    '''
    最终准确率测试
    '''
    print('每5个epoch的准确率为:')
    for i in res:
        print(i)
    print('*****************开始测试*****************')
    total_correct = 0
    total_testset = 0
    for x, label in test_loader:
        x, label = x.permute(0, 2, 1).to(device), label.to(device)  # x数据是BxNX3 变为 Bx3xN
        out = net(x)
        pred_choice = out.data.max(1)[1]
        correct = pred_choice.eq(label.data).cpu().sum()
        total_correct += correct.item()
        total_testset += x.size()[0]
    print("final accuracy {}".format(total_correct / float(total_testset)))


if __name__ == '__main__':
    main()
