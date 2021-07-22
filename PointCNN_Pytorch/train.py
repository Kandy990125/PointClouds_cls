import numpy as np
import torch
from torch import nn

from PointCNN_Pytorch import provider
from PointCNN_Pytorch.model.classifier import Classifier
from dataset.ModelNet40 import get_data_loader
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
blue = lambda x: '\033[94m' + x + '\033[0m'

batch_size = 32
num_epochs = 10
train_loader = get_data_loader(train=True, batch_size=batch_size)
test_loader = get_data_loader(train=False, batch_size=batch_size)
print("----------already load datasets----------")
model = Classifier(NUM_CLASS=40).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
res = []
for epoch in range(num_epochs):
    for x, label in train_loader:
        # x = x.permute(0, 2, 1).to(device)
        # print(x)
        label = label.to(device)
        label = Variable(label, requires_grad=False).cuda()

        rotated_data = provider.rotate_point_cloud(x)
        jittered_data = provider.jitter_point_cloud(rotated_data)  # P_Sampled
        P_sampled = jittered_data
        F_sampled = np.zeros((batch_size, 2048, 0))
        optimizer.zero_grad()
        P_sampled = torch.from_numpy(P_sampled).float()
        P_sampled = Variable(P_sampled, requires_grad=False).cuda()

        out = model((P_sampled, P_sampled))
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()
        pred_choice = out.data.max(1)[1]
        print("result:"+str(pred_choice))
        correct = pred_choice.eq(label.data).cpu().sum()
        # print('[%d: %d/%d] train loss: %f accuracy: %f' %
        # (epoch, idx, num_batch, loss.item(), correct.item() / float(batch_size)))
        print('[%d] train loss: %f accuracy：%f' % (epoch, loss.item(), correct.item() / float(batch_size)))

        if (epoch+1) % 5 == 0:
            total_correct = 0
            total_testset = 0
            for x, label in train_loader:
                rotated_data = provider.rotate_point_cloud(x)
                jittered_data = provider.jitter_point_cloud(rotated_data)  # P_Sampled
                P_sampled = jittered_data
                F_sampled = np.zeros((batch_size, 2048, 0))
                P_sampled = torch.from_numpy(P_sampled).float()
                P_sampled = Variable(P_sampled, requires_grad=False).cuda()
                label = label.to(device)
                label = Variable(label, requires_grad=False).cuda()
                out = model((P_sampled, P_sampled))
                pred_choice = out.data.max(1)[1]
                correct = pred_choice.eq(label.data).cpu().sum()
                total_correct += correct.item()
                total_testset += x.size()[0]
            print("The {} is {}.\t Test accuracy {}".format(blue('epoch'), epoch, total_correct / float(total_testset)))
            res.append(total_correct / float(total_testset))
            torch.save(model.state_dict, 'out/model_epoch%d.pth' % (epoch))
