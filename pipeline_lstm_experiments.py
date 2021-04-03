from sklearn.preprocessing import StandardScaler
from torch import nn
import torch
import numpy as np


rnn = nn.LSTM(input_size=5, hidden_size=10, num_layers=4, dropout=0)
input = torch.randn((2, 1, 5), dtype=torch.float32)

scaler = StandardScaler()

f = np.random.randn(8, 2)
f2 = np.random.randn(2, 2)
scaler.fit(f)
print(f)
print(f2)

new_input1 = scaler.transform(f)
new_input2 = scaler.transform(f2)
print('1: ', new_input1)
print('2: ', new_input2)

h0 = torch.zeros((4, 1, 10), dtype=torch.float32)
c0 = torch.zeros((4, 1, 10), dtype=torch.float32)

output, (hn, cn) = rnn(input, (h0, c0))

lin = nn.Linear(in_features=10, out_features=1)

print('input: ', input, 'shape: ', input.shape)
print('output: ', output, 'shape: ', output.shape)
print('output linear: ', lin(output), 'shape: ', output.shape)
print('output linear: ', torch.flatten(lin(output)), 'shape: ', output.shape)
