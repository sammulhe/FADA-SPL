"""
@author: Mingyang Liu
@contact: mingyang1024@gmail.com
"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import warnings
from configs.data_model_configs import UCIHAR
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd

class Load_Dataset(Dataset):
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train).long()

        # (channel, length)
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        
        if X_train.shape.index(min(X_train.shape[1], X_train.shape[2])) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train
        self.y_data = y_train

        self.num_channels = X_train.shape[1]

        # normalize
        data_mean = torch.mean(X_train, dim=(0, 2))
        data_std = torch.std(X_train, dim=(0, 2))
        self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None

        #return self.x_data[index].float(), self.y_data[index].long()
        return x, y

    def __len__(self):
        return self.len


class har_data(Dataset):
    def __init__(self, filepath):


        self.data = pd.read_csv(filepath)

        self.data = self.data.values
        self.X, self.Y = self.data[:, 1:], self.data[:, 0]
        self.X = torch.FloatTensor(self.X)
        self.X = self.X.reshape(self.X.shape[0], 3, 128)
        self.Y = torch.LongTensor(self.Y)

    def __len__(self):
        return self.X.shape[0]


    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def get_dim(self):
        return self.X.shape[1]


def data_generator(data_path, domain_id, args):
    # loading path
    train_dataset = torch.load(os.path.join('.' + data_path, "train_" + domain_id + ".pt"), weights_only=False)
    test_dataset = torch.load(os.path.join('.' + data_path, "test_" + domain_id + ".pt"), weights_only=False)

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset)
    test_dataset = Load_Dataset(test_dataset)
    # train_dataset = har_data(os.path.join('.' + data_path, "ucihhar_" + domain_id + "_train.csv"))
    # test_dataset = har_data(os.path.join('.' + data_path, "ucihhar_" + domain_id + "_test.csv"))
    batch_size = args.bs
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=args.shuffle, drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=True)
    return train_loader, test_loader


#data_generator('/data/UCIHAR', '2', None)


