import torch
import torch.nn as nn
import torch.optim as optim
import custom_rnn as rnn


class CustomRNN(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        self.U_g = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_g = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_g = nn.Parameter(torch.Tensor(hidden_sz))

        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

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
            h_t = torch.sigmoid(x_t @ self.U_g + h_t @ self.W_g + self.b_g)
            h_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.W_o + self.b_o)
            h_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.W_i + self.b_i)

            hidden_seq.append(h_t.unsqueeze(
                1))  # transform h_t from shape (batch_size, hidden_size) to shape (batch_size, 1, hidden_size)

        # reshape hidden_seq
        hidden_seq = torch.cat(hidden_seq,
                               dim=1)  # concatenate list of tensors into one tensor along dimension 1 (batch_size,
        # sequence_size, hidden_size)
        return h_t, hidden_seq


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = rnn.CustomLSTM(4, 5)  # input_size, hidden_size
        self.layer_2 = rnn.CustomLSTM(5, 10)  # input_size, hidden_size
        self.layer_3 = nn.Linear(10, 1)  # input_size, output_size

    def forward(self, x):
        out, hidden = self.layer_1(x)  # returns tuple consisting of output and sequence
        out, hidden = self.layer_2(hidden)
        output = torch.relu(self.layer_3(out))
        return output


net = Net()

inputs = torch.randn(10, 20, 4)  # batch_size, seq_size, input_size
labels = torch.randn(10)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(6):
    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(torch.squeeze(outputs), labels)
    loss.backward()
    optimizer.step()
    print(loss.item())


class CustomLSTM2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weights_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        h_prev, c_prev = hidden
        combined = torch.cat((h_prev, x), dim=1)
        gates = torch.mm(combined, torch.cat((self.weights_ih, self.weights_hh), dim=1).t()) + self.bias
        i, f, g, o = gates.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)





