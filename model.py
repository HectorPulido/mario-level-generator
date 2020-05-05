import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, device, n_layers=3):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device
        self.rnn = nn.LSTM(input_size, hidden_size, self.n_layers,
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, hidden, x):

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(self.device))

        return hidden

    def one_hot_encode(self, data, categories):
        one_hot = []
        for d in data:
            one = [0 for _ in range(categories)]
            one[d] = 1
            one_hot.append(one)
        return torch.tensor(one_hot).to(self.device).float()

    def input_process(self, data, chars_to_indx):
        data = [chars_to_indx[d] for d in data]
        return self.one_hot_encode(data, self.input_size)

    def output_process(self, data, chars_to_indx):
        data = [chars_to_indx[d] for d in data]
        data = data[1:]
        data.append(self.input_size - 1)
        return torch.tensor(data).to(self.device)

