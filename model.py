import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from math import sin, cos
import torch.nn.functional as F
import torch.optim as optim
import time


# A LSTM model
class LSTM_news_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(LSTM_news_classifier, self).__init__()
        self.name = "LSTM_simple"
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, (h_n, c_n) = self.rnn(x, (h0, c0))
        return self.fc(out[:,-1,:])

# A bidirectional LSTM.
class LSTM_news_classifier_bidirectional(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(LSTM_news_classifier_bidirectional, self).__init__()
        self.name = "LSTM_bidirectional"
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, num_class)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size)
        c0 = torch.zeros(2, x.size(0), self.hidden_size)
        out, (h_n, c_n) = self.rnn(x, (h0, c0))
        return self.fc(out[:,-1,:])

# Positional encoding for transformer
class positional_encoding(nn.Module):
    def __init__(self, max_length, embedding_size):
        super(positional_encoding, self).__init__()
        self.pe_tensor = torch.zeros(max_length, embedding_size)
        for pos in range(max_length):
            for i in range(embedding_size):
                if i % 2 == 0:
                    pe = sin(pos / pow(10000, (2 * i / max_length)))
                else:
                    pe = cos(pos / pow(10000, (2 * (i - 1) / max_length)))
                self.pe_tensor[pos][i] = pe

    def forward(self, x):
        pe_input = self.pe_tensor[:x.shape[1], :]
        pe_input = pe_input.repeat(x.shape[0], 1, 1)
        x = x + pe_input
        return x

# Transformer (encoder only)
class Transformer_news_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(Transformer_news_classifier, self).__init__()
        self.pos_encoding = positional_encoding(1000, input_size)
        self.name = "Transformer_news_classifier_2"
        self.linear_q = nn.Linear(input_size, hidden_size)
        self.linear_k = nn.Linear(input_size, hidden_size)
        self.linear_v = nn.Linear(input_size, hidden_size)
        self.linear_x = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # print(x.shape)
        x = x + self.pos_encoding(x)
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x = self.norm(self.linear_x(x) + self.attention(q, k, v)[0])
        x = self.norm(x + self.fc1(x))
        x = torch.sum(x, 1)
        return self.fc2(x)