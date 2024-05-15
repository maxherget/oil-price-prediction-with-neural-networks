import torch
import torch.nn as nn
import math


class CustomRNN(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        bs, seq_sz, _ = x.shape  # assumes x.shape represents (batch_size, sequence_size, input_size)
        hidden_seq = []
        h_t = torch.zeros(bs, self.hidden_size)  # initialize states

        for t in range(seq_sz):
            x_t = x[:, t, :]
            h_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.W_f + self.b_f)

            hidden_seq.append(h_t.unsqueeze(1))  # transform h_t from shape (batch_size, hidden_size) to shape (batch_size, 1, hidden_size)

        # reshape hidden_seq
        hidden_seq = torch.cat(hidden_seq,dim=1)  # concatenate list of tensors into one tensor along dimension 1 (batch_size, sequence_size, hidden_size)
        return h_t, hidden_seq