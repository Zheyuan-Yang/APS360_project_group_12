import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from math import sin, cos
import torch.nn.functional as F
import torch.optim as optim
import time


# LSTM model
class LSTM_news_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(LSTM_news_classifier, self).__init__()
        self.name = "LSTM_1"
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, (h_n, c_n) = self.rnn(x, (h0, c0))
        return self.fc(out[:,-1,:])

# LSTM model number 2. I add a sigmoid function
class LSTM_news_classifier_2(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(LSTM_news_classifier_2, self).__init__()
        self.name = "LSTM_2"
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)
        self.af = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, (h_n, c_n) = self.rnn(x, (h0, c0))
        return self.af(self.fc(out[:,-1,:]))

# I made this a bidirectional LSTM.
class LSTM_news_classifier_3(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(LSTM_news_classifier_3, self).__init__()
        self.name = "LSTM_3"
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, num_class)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size)
        c0 = torch.zeros(2, x.size(0), self.hidden_size)
        out, (h_n, c_n) = self.rnn(x, (h0, c0))
        return self.fc(out[:,-1,:])


class LSTM_news_classifier_4(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(LSTM_news_classifier_4, self).__init__()
        self.name = "LSTM_4"
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True, num_layers=4)
        self.fc = nn.Linear(4 * 2 * hidden_size, num_class)

    def forward(self, x):
        h0 = torch.zeros(8, x.size(0), self.hidden_size)
        c0 = torch.zeros(8, x.size(0), self.hidden_size)
        out, (h_n, c_n) = self.rnn(x, (h0, c0))
        return self.fc(h_n.view(-1, self.hidden_size * 4 * 2))

class Transformer_news_classifier_2(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(Transformer_news_classifier_2, self).__init__()
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
        for batch in range(x.shape[0]):
            for pos in range(x.shape[1]):
                for i in range(x.shape[2]):
                    if i % 2 == 0:
                        pe = sin(pos / pow(10000, (2 * i / (x.shape[2]))))
                    else:
                        pe = cos(pos / pow(10000, (2 * (i - 1) / (x.shape[2]))))
                    x[batch][pos][i] = x[batch][pos][i] + pe
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x = self.norm(self.linear_x(x) + self.attention(q, k, v)[0])
        x = self.norm(x + self.fc1(x))
        x = torch.sum(x, 1)
        return self.fc2(x)