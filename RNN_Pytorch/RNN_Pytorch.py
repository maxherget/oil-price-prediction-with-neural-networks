import torch
import torch.nn as nn
import torch.optim as optim
import custom_rnn as rnn

class HTWNet(nn.Module):

    def __init__(self):
        super(HTWNet, self).__init__()
        self.layer_1 = rnn.CustomRNN(3, 5) # input_size, hidden_size
        self.layer_2 = rnn.CustomRNN(5, 10) # input_size, hidden_size
        self.layer_3 = nn.Linear(10, 1) # input_size, output_size

    def forward(self, x):
        out, hidden = self.layer_1(x) # returns tuple consisting of output and sequence
        out, hidden = self.layer_2(hidden)
        output = torch.relu(self.layer_3(out))
        return output

net = HTWNet()

inputs = torch.randn(20, 10, 3) # batch_size, seq_size, input_size
labels = torch.randn(20)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

loss_vals = list()
for epoch in range(100):

    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(torch.squeeze(outputs), labels) # torch.squeeze removes all dimensions of size=1
    loss.backward()
    optimizer.step()
    print(loss.item())
    loss_vals.append(loss.item())

import matplotlib.pyplot as plt
plt.plot(range(100), loss_vals)
print(loss_vals[99])