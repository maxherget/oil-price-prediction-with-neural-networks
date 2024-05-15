import torch
import torch.nn as nn
import math


class CustomLinear(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz)) # https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter
        self.b = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) # Xavier initialization
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv) # https://pytorch.org/docs/stable/torch.html#inplace-random-sampling

    def forward(self, x):
        out = x @ self.W + self.b # https://pytorch.org/docs/stable/generated/torch.matmul.html
        return out