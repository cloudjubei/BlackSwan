
import torch
from torch.utils.data import Dataset

def create_lookback_dataset(features, labels, lookback):
    X, y = [], []
    for i in range(len(features)-lookback-1):
        feature = features[i:i+lookback]
        # label = labels[i+lookback+1]
        label = labels[i+lookback-1]
        X.append(feature)
        y.append(label)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LookbackDataset(Dataset):
    def __init__(self, timeseries_features, timeseries_labels, lookback=2):
        X, y = create_lookback_dataset(timeseries_features, timeseries_labels, lookback)
        self.X = X
        self.y = y
        print(f'X shape: {X.shape}')
        print(f'y shape: {y.shape}')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TimeseriesDataset(Dataset):
    def __init__(self, timeseries_features, timeseries_labels):
        # X, y = create_lookback_dataset(timeseries_features, timeseries_labels, lookback)
        self.X = torch.tensor(timeseries_features, dtype=torch.float32)
        self.y = torch.tensor(timeseries_labels, dtype=torch.float32)
        # print(f'X shape: {X.shape}')
        # print(f'y shape: {y.shape}')

    def __len__(self):
        return 1
        # return len(self.X)

    def __getitem__(self, idx):
        return self.X, self.y
        # return self.X[idx], self.y[idx]
    