from sklearn.preprocessing import StandardScaler
from torch import nn
import torch
import pandas as pd
import numpy as np

data_path = 'C:/Users/Zuzu/Desktop/tcc/fullPipeline/lstm_data/sentiment_classification_twitter_100_stemmed.p'
labels_path = 'C:/Users/Zuzu/Desktop/tcc/fullPipeline/lstm_data/converted_labels_twitter_100_stemmed.p'

n_users = 20000
seq_len = 5
train_prop = 0.7

dataset = pd.read_pickle(data_path)
labels = pd.read_pickle(labels_path)

merged_data = {
    'x': [],
    'y': []
}

train_data = {
    'x': [],
    'y': []
}

test_data = {
    'x': [],
    'y': []
}

for i in range(0, len(dataset)):
    if(len(dataset[i]) >= seq_len):
        merged_data['x'].append(dataset[i])
        merged_data['y'].append(labels[i])

for i in range(0, len(merged_data['x'])):
    if (i + 1) / len(merged_data['x']) <= train_prop:
        train_data['x'].append(merged_data['x'][i])
        train_data['y'].append(merged_data['y'][i])
    else:
        test_data['x'].append(merged_data['x'][i])
        test_data['y'].append(merged_data['y'][i])

rnn = nn.LSTM(input_size=5, hidden_size=10, num_layers=4, dropout=0).cuda()

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
