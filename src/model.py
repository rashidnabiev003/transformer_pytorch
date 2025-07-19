import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings() * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:int):
        super().__init__()
        self.d_model = d_model
        self.seq_leng = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of positional encodings
        pe = torch.zeros(seq_len, d_model) # shape (seq_len, d_model)
        # Calculate the positional encodings
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # shape (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) 

        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :].float()).require_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # batch size, seq_len, d_model --> batch size, seq_len, d_ff
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_module:int, h:int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_module
        self.h = h
        assert d_module % h == 0, "d_model must be divisible by h"

        self.d_k = d_module // h
        self.w_q = nn.Linear(d_module, d_module)
        self.w_k = nn.Linear(d_module, d_module)
        self.w_v = nn.Linear(d_module, d_module)
        self.w_o = nn.Linear(d_module, d_module)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # (batch_size, h, seq_len, seq_len)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch_size, h, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return attention_scores @ value  
        

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape for multi-head attention
        query = query.viev(query.shape(0), query.shape(1), self.h, self.d_k).transpose(1, 2)  # (batch_size, seq_len, h, d_k)
        key = key.view(key.shape(0), key.shape(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_len, h, d_k)
        value = value.view(value.shape(0), value.shape(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_len, h, d_k)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape(0), self.h * self.d_k)

        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    