from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

class LSTMLinear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, scaler: StandardScaler):
        super().__init__()
        self.scaler = scaler

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (ht, _) = self.lstm(x)
        out = ht[-1]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)

        if len(out.size()) > 1: 
            out = torch.squeeze(out, 1)
        return out