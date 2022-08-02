import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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
