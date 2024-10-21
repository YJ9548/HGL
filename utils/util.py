from torch_geometric.data import Data, Batch
import torch.utils.data as data
import copy
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import torch
import torch_geometric

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, max, min):
        self.max = max
        self.min = min
    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

class NetDataSet(data.Dataset):
    def __init__(self, x, y, A, isTrainSet=False):
        self.isTrainSet = isTrainSet
        self.x = x.astype('float32')
        self.y = y.astype('float32')
        self.A = A.astype('float32')

    def __getitem__(self, index):
        if self.isTrainSet:
            return self.x[index].transpose(1, 0, 2), self.y[index].transpose(2, 1, 0), self.x_[index].transpose(1, 0, 2)
        else:
            return self.x[index], self.y[index], self.A[index]

    def __len__(self):
        return self.x.shape[0]

class NetDataSet_f(data.Dataset):
    def __init__(self, x, x_f, y, A, A_f, isTrainSet=False):
        self.isTrainSet = isTrainSet
        self.x = x.astype('float32')
        self.x_f = x_f.astype('float32')
        self.y = y.astype('float32')
        self.A = A.astype('float32')
        self.A_f = A_f.astype('float32')

    def __getitem__(self, index):
        if self.isTrainSet:
            return self.x[index].transpose(1, 0, 2), self.y[index].transpose(2, 1, 0), self.x_[index].transpose(1, 0, 2)
        else:
            return self.x[index],self.x_f[index], self.y[index], self.A[index], self.A_f[index]

    def __len__(self):
        return self.x.shape[0]

def DataLoader(x, y, A, batch_size, isTrainSet=False, shuffle=True):
    dataset = NetDataSet(x, y, A, isTrainSet)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def DataLoader_f(x, x_f, y, A, A_f, batch_size, isTrainSet=False, shuffle=True):
    dataset = NetDataSet_f(x, x_f, y, A, A_f, isTrainSet)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def dense_to_ind_val(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)
    index = (torch.isnan(adj) == 0).nonzero(as_tuple=True)
    edge_attr = adj[index]
    return torch.stack(index, dim=0), edge_attr

def to_dense(data):
    '''Returns a copy of the data object in Dense form.
    '''
    denser = torch_geometric.transforms.ToDense()
    copy_data = denser(copy.deepcopy(data))
    return copy_data

# 生成 batch 图数据
def graph_data(fmri, corr):
    data_list = []

    for i in range(corr.size()[0]):
        edge_index, edge_attr = dense_to_ind_val(corr[i])
        data_list.append(Data(x=corr[i], edge_index=edge_index, edge_attr=edge_attr))

    graph_batch = Batch.from_data_list(data_list)
    return graph_batch

def inverse_transform(data, scaler):
    max = scaler.max
    min = scaler.min
    max = torch.unsqueeze(max, 0)
    min = torch.unsqueeze(min, 0)

    data = data * (max - min) + min

    return data

def load_dataset_cpm(pop="NT"):
    '''Loads the data for given population in the upper triangular matrix form
    as required by CPM functions.
    '''
    connectomes = np.array(torch.load(f"connectome_{pop}.ts"))
    fiq_scores = np.array(torch.load(f"fiq_{pop}.ts"))
    viq_scores = np.array(torch.load(f"viq_{pop}.ts"))

    fc_data = {}
    behav_data = {}
    for subject in range(fiq_scores.shape[0]):  # take upper triangular part of each matrix
        fc_data[subject] = connectomes[:, :, subject][np.triu_indices_from(connectomes[:, :, subject], k=1)]
        behav_data[subject] = {'fiq': fiq_scores[subject].item(), 'viq': viq_scores[subject].item()}
    return pd.DataFrame.from_dict(fc_data, orient='index'), pd.DataFrame.from_dict(behav_data, orient='index')


def get_folds(data_list, k_folds=5):
    '''Divides a data list into lists
       with k elements such that each element
       is the data used in that cross validation fold
    '''
    train_folds, test_folds = [], []
    for train_idx, test_idx in KFold(k_folds, shuffle=False, random_state=None).split(data_list):
        train_folds.append([data_list[i] for i in train_idx])
        test_folds.append([data_list[i] for i in test_idx])
    return train_folds, test_folds


def get_loaders(train, test, batch_size=1):
    '''Returns data loaders for given data lists
    '''
    train_loader = torch_geometric.data.DataLoader(train, batch_size=batch_size)
    test_loader = torch_geometric.data.DataLoader(test, batch_size=batch_size)
    return train_loader, test_loader


def load_dataset_tensor(pop="NT"):
    '''Loads dataset as tuple of (tensor of connectomes,
       tensor of fiq scores, tensor of viq scores)
    '''
    connectomes = torch.load(f"connectome_{pop}.ts")
    fiq_scores = torch.load(f"fiq_{pop}.ts")
    viq_scores = torch.load(f"viq_{pop}.ts")
    return connectomes, fiq_scores, viq_scores


def to_dense(data):
    '''Returns a copy of the data object in Dense form.
    '''
    denser = torch_geometric.transforms.ToDense()
    copy_data = denser(copy.deepcopy(data))
    return copy_data