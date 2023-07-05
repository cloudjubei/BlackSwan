import torch
import torch.nn as nn

class SpatialDropout1D(nn.Module):
    def __init__(self, p):
        super(SpatialDropout1D, self).__init__()
        self.dropout = nn.Dropout3d(p=p)

    def forward(self, x):
        # Expand dimensions to match the expected input shape of Dropout2d
        x = x.unsqueeze(2)
        x = self.dropout(x)
        # Squeeze the additional dimension to return to 1D
        x = x.squeeze(2)
        return x
    

class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query_layer = nn.Linear(query_dim, key_dim)
        self.key_layer = nn.Linear(key_dim, key_dim)
        self.value_layer = nn.Linear(key_dim, value_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys):
        query = self.query_layer(query)
        keys = self.key_layer(keys)
        scores = torch.matmul(query, keys.transpose(-2, -1))
        attention_weights = self.softmax(scores)
        weighted_values = torch.matmul(attention_weights, self.value_layer(keys))
        return weighted_values, attention_weights