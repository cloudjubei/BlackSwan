from torch import nn
import torch

class LSTMLocal(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, with_hidden=True):
        super(LSTMLocal, self).__init__()
        self.with_hidden = with_hidden
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=False)
        self.hidden = (torch.zeros(num_layers, hidden_size),torch.zeros(num_layers,hidden_size))

    def forward(self, x):
        x, (h1, h2) = self.lstm(x, self.hidden)
        if self.with_hidden:
            self.hidden = (h1.detach(), h2.detach())
        return x
class GRULocal(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, with_hidden=True):
        super(GRULocal, self).__init__()
        self.with_hidden = with_hidden
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=False)
        self.hidden = torch.zeros(num_layers, hidden_size)

    def forward(self, x):
        x, h1 = self.gru(x, self.hidden)
        if self.with_hidden:
            self.hidden = h1.detach()
        return x