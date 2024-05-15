# https://pytorch.org/docs/stable/index.html
# --> torch.nn --> Containers --> Module
import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):  # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module

    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(3, 3)  # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        self.layer_2 = nn.Linear(3, 3)
        self.output_layer = nn.Linear(3, 2)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.sigmoid(self.layer_2(x))
        x = self.output_layer(x)
        return x


net = Net()

# The weights of the neural network have been initialized:
print(net.layer_1.weight)
print(net.layer_2.weight)
print(net.output_layer.weight)

# https://pytorch.org/docs/stable/tensors.html
inputs = torch.randn(20, 3)
labels = torch.randn(20, 2)

# https://pytorch.org/docs/stable/nn.html#loss-functions
criterion = nn.MSELoss()

# https://pytorch.org/docs/stable/optim.html
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Forward step is executed in Net class.
outputs = net(inputs)
print(outputs)
print(labels)

# Loss is calculated.
loss = criterion(outputs, labels)
print(loss.item())

# If any input Tensor of an operation has 'requires_grad=True', the computation will be tracked
# (e.g. the initialized weight matrices above).
# See also here: https://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial.html
print(loss.grad_fn.next_functions)

# Backward propagation of error:
loss.backward()

#Weights are updated:
optimizer.step()
print(net.layer_1.weight)
print(net.layer_2.weight)

# Loop over dataset multiple times. Pytorch accumulates gradients per default, so it is required to
# zero the parameter gradients before each iteration with 'zero_grad()' function.
loss_vals = list()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    loss_vals.append(loss.item())

# Plot loss:
import matplotlib.pyplot as plt

plt.plot(range(100), loss_vals)
plt.show()
