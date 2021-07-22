import h5py
import torch
from torch.utils.data import Dataset, DataLoader
main_path = "../data/modelnet40_ply_hdf5_2048/"
train_txt_path = main_path + "train_files.txt"
valid_txt_path = main_path + "test_files.txt"


def get_data(train=True):
    data_txt_path = train_txt_path if train else valid_txt_path
    with open(data_txt_path, "r") as f:
        txt = f.read().splitlines()
    clouds_li = []
    labels_li = []
    for file_name in txt:
        file_name = "../" + file_name
        h5 = h5py.File(file_name)
        # for key in h5.keys():
        #   print(h5[key], key, h5[key].name)
        lbl = h5["label"]
        # print(torch.Tensor(lbl))
        pts = h5["data"]

        clouds_li.append(torch.Tensor(pts))
        labels_li.append(torch.Tensor(lbl))
    clouds = torch.cat(clouds_li)
    # print(clouds)
    labels = torch.cat(labels_li)
    # print(labels.long().squeeze())
    return clouds, labels.long().squeeze()


class ModelNet40(Dataset):
    def __init__(self, train=True):
        clouds, labels = get_data(train=train)

        self.x_data = clouds
        self.y_data = labels
        self.length = clouds.size(0)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def get_data_loader(train=True, batch_size=32):
    point_data_set = ModelNet40(train=True)
    point_data_loader = DataLoader(dataset=point_data_set, batch_size=batch_size, shuffle=train)
    return point_data_loader
