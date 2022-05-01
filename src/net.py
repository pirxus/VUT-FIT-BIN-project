import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_dim = 8 # 8 observation inputs
        self.out_dim = 4 # 4 action outputs

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.fc1 = nn.Linear(self.in_dim, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 4)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return self.softmax(out)


def relu(x: np.ndarray) -> np.ndarray:
    return x * (x > 0)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))

class NpNet:
    def __init__(self, input_dim=8, hidden_dim_1=6, hidden_dim_2=5, out_dim=4, zero_bias=False):
        self.num_layers = 4
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.out_dim = out_dim

        # init the network weights and biases. The chosen architecture is [8, 6, 6, 4]
        self.w1 = np.random.randn(self.hidden_dim_1, self.input_dim) * np.sqrt((2 / self.input_dim))
        self.w2 = np.random.randn(self.hidden_dim_2, self.hidden_dim_1) * np.sqrt((2 / self.hidden_dim_1))
        self.w3 = np.random.randn(self.out_dim, self.hidden_dim_2) * np.sqrt((1 / self.hidden_dim_2))

        if zero_bias:
            self.b1 = np.zeros(self.hidden_dim_1)
            self.b2 = np.zeros(self.hidden_dim_2)
            self.b3 = np.zeros(self.out_dim)
        else:
            self.b1 = np.random.randn(self.hidden_dim_1) * np.sqrt((2 / self.input_dim))
            self.b2 = np.random.randn(self.hidden_dim_2) * np.sqrt((2 / self.hidden_dim_1))
            self.b3 = np.random.randn(self.out_dim) * np.sqrt((1 / self.hidden_dim_2))


    def forward(self, x, output_only=True):

        out1 = np.dot(self.w1, x) + self.b1
        out = relu(out1)

        out2 = np.dot(self.w2, out) + self.b2
        out = relu(out2)

        out = np.dot(self.w3, out) + self.b3

        if not output_only:
            return out, out1, out2
        return out
