from sklearn.preprocessing import StandardScaler
from torch import nn
import torch
import numpy as np

print(torch.cuda.is_available())

rnn = nn.LSTM(input_size=5, hidden_size=10, num_layers=4, dropout=0).cuda()

n_users = 20000
seq_len = 10

input = np.random.randn(n_users, seq_len, 5)
scaler = StandardScaler()
fit_data = input.reshape((n_users*seq_len, 5))
scaler.fit(fit_data)

new_input1 = []

for arr in input:
    scaled = scaler.transform(arr)
    new_input1.append(scaled)

new_input1 = torch.from_numpy(np.array(new_input1, dtype=np.float32)).cuda()
print(new_input1.device)

h0 = torch.zeros((4, seq_len, 10), dtype=torch.float32).cuda()
c0 = torch.zeros((4, seq_len, 10), dtype=torch.float32).cuda()

output, (hn, cn) = rnn(new_input1, (h0, c0))

result = rnn(new_input1)

sigmoid = nn.Sigmoid().cuda()
linear = nn.Linear(in_features=10, out_features=1).cuda()

print('input: ', new_input1, 'shape: ', new_input1.shape)
print('output: ', output, 'shape: ', output.shape)

tweet_layer = torch.squeeze(sigmoid(linear(output)), axis=2).cuda()
print('output sigmoid per tweet: ', tweet_layer, 'shape: ', tweet_layer.shape)
puser = torch.flatten(sigmoid(linear(tweet_layer))).cuda()
print('output sigmoid per user: ', puser, 'shape: ', puser.shape)

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss().cuda()
print('hueidhfuidshuih', result[0].shape)

loss = loss_fn(result[0].float().cuda(), torch.zeros((n_users, 10)).long().cuda())

print(loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()
