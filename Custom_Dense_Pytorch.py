import torch
import torch.nn as nn
import torch.optim as optim
import custom_dense as cd


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = cd.CustomLinear(3, 3)
        self.layer_2 = cd.CustomLinear(3, 3)
        self.output_layer = cd.CustomLinear(3, 2)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.sigmoid(self.layer_2(x))
        x = torch.softmax(self.output_layer(x), dim=1)
        return x

net = Net()

inputs = torch.randn(20,3)
labels = torch.randn(20,2)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(100):

    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(loss.item())