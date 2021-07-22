import torch
from torch import nn
from PointCNN_Pytorch.model.model import *
from PointCNN_Pytorch.utils.util_funcs import knn_indices_func_gpu

AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)

class Classifier(nn.Module):

    def __init__(self, NUM_CLASS):
        super(Classifier, self).__init__()

        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, NUM_CLASS, with_bn=False, activation=None)
        )

    def forward(self, x):
        x = self.pcnn1(x)
        # if False:
        #   print("Making graph...")
        #   k = make_dot(x[1])
        #
        #   print("Viewing...")
        #    k.view()
        #   print("DONE")
        #
        #   assert False
        x = self.pcnn2(x)[1]  # grab features

        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean