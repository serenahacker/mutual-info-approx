import torch
import torch.nn as nn


class QPhi(nn.Module):
    def __init__(self, x_concat_u_size, hidden_size=50, z_size=10):
        super(QPhi, self).__init__()
        self.fc1 = nn.Linear(x_concat_u_size, hidden_size)
        self.activation = nn.Softplus()
        self.fc2 = nn.Linear(hidden_size, z_size)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x