from sklearn.preprocessing import StandardScaler
from torch import nn
import torch
import numpy as np


rnn = nn.LSTM(input_size=5, hidden_size=10, num_layers=4, dropout=0)

input = np.random.randn(2, 10, 5)
scaler = StandardScaler()
fit_data = input.reshape((20, 5))
scaler.fit(fit_data)

new_input1 = []

for arr in input:
    scaled = scaler.transform(arr)
    new_input1.append(scaled)

new_input1 = torch.from_numpy(np.array(new_input1, dtype=np.float32))

h0 = torch.zeros((4, 10, 10), dtype=torch.float32)
c0 = torch.zeros((4, 10, 10), dtype=torch.float32)

output, (hn, cn) = rnn(new_input1, (h0, c0))

sigmoid = nn.Sigmoid()
linear = nn.Linear(in_features=10, out_features=1)

print('input: ', new_input1, 'shape: ', new_input1.shape)
print('output: ', output, 'shape: ', output.shape)

tweet_layer = torch.squeeze(sigmoid(linear(output)), axis=2)
print('output sigmoid per tweet: ', tweet_layer, 'shape: ', tweet_layer.shape)
puser = torch.flatten(sigmoid(linear(tweet_layer)))
print('output sigmoid per user: ', puser, 'shape: ', puser.shape)
