
from torch import nn
import torch as th
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        batch_size, seq_len = x.size()
        query = self.query(x)  # (batch_size, seq_len, hidden_dim)
        key = self.key(x)      # (batch_size, seq_len, hidden_dim)
        value = self.value(x)  # (batch_size, seq_len, hidden_dim)

        q = query.view(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        k = key.view(batch_size, seq_len, -1)    # (batch_size, seq_len, hidden_dim)
        v = value.view(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_dim)

        scores = th.bmm(q, k.transpose(1, 2)) / th.sqrt(th.tensor(q.size(-1), dtype=th.float32))  # (batch_size, seq_len, seq_len)
        attn_weights = self.softmax(scores)  # (batch_size, seq_len, seq_len)
        attn_output = th.bmm(attn_weights, v)  # (batch_size, seq_len, hidden_dim)

        output = self.out(attn_output.squeeze(2))  # (batch_size, seq_len, in_dim)
        return output
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        scores = th.matmul(query, key.transpose(-2, -1)) / th.sqrt(th.tensor(self.hidden_dim, dtype=th.float32))
        attn_weights = self.softmax(scores)
        output = th.matmul(attn_weights, value)
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.attention = th.nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=False)
        
    def forward(self, x):
        output, weights = self.attention(x, x, x)
        return output
    
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        query_with_time_axis = x.unsqueeze(1)
        score = self.V(th.tanh(self.W1(query_with_time_axis) + self.W2(x)))
        attention_weights = th.softmax(score, dim=1)
        context_vector = attention_weights * x
        context_vector = th.sum(context_vector, dim=1)
        return context_vector
    
class GlobalContextAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(GlobalContextAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)

    def forward(self, x):
        # batch_size, seq_len = x.size()
        query = self.query(x).transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        key = self.key(x)  # (batch_size, seq_len, hidden_dim)
        value = self.value(x)  # (batch_size, seq_len, hidden_dim)

        scores = th.bmm(query.transpose(1, 2), key)  # (batch_size, seq_len, seq_len)
        attn_weights = th.nn.functional.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        context = th.bmm(attn_weights, value)  # (batch_size, seq_len, hidden_dim)
        context = context.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        context = self.conv(context)  # (batch_size, hidden_dim, seq_len)
        context = context.mean(dim=-1)  # (batch_size, hidden_dim)
        output = x * context.unsqueeze(1)  # (batch_size, seq_len, hidden_dim)
        return output

    