from sklearn.preprocessing import StandardScaler
from torch import nn
import torch
import pandas as pd
import numpy as np
import itertools
import tqdm
from lstm.LSTMHelper import LSTMHelper

data_path = '../lstm_data/sentiment_classification_twitter_100_stemmed.p'
labels_path = '../lstm_data/converted_labels_twitter_100_stemmed.p'

seq_len = 5
train_prop = 0.7
input_size = 2
_hidden_size = 4
n_layers = 4

dataset = pd.read_pickle(data_path)
labels = pd.read_pickle(labels_path)

x_all, y_all = LSTMHelper.create_sequences(dataset, labels, seq_len)
rnn = nn.LSTM(input_size=input_size, hidden_size=_hidden_size, num_layers=n_layers, dropout=0.5).cuda()

scaler = StandardScaler()
flatten = list(itertools.chain(*dataset))
scaler.fit(flatten)

new_input1 = []

for user_corpus in tqdm.tqdm(x_all[:1]):
    tweet_sequences = []
    for sequence in user_corpus:
        tweet_sequences.append(scaler.transform(sequence))
    new_input1.append(tweet_sequences)
    
new_input1 = [torch.from_numpy(np.array(user_corpus, dtype=np.float32)).cuda() for user_corpus in new_input1]

h0 = torch.zeros((n_layers, seq_len, _hidden_size), dtype=torch.float32).cuda()
c0 = torch.zeros((n_layers, seq_len, _hidden_size), dtype=torch.float32).cuda()

output, (hn, cn) = rnn(new_input1[0], (h0, c0))
print('output: ', output, 'shape: ', output.shape)

sigmoid = nn.Sigmoid().cuda()


linear_hidden = nn.Linear(in_features=_hidden_size, out_features=1).cuda()
sig = sigmoid(linear_hidden(output))
tweet_layer = torch.squeeze(sig, axis=2).cuda()
print('output sigmoid per tweet: ', tweet_layer, 'shape: ', tweet_layer.shape)


linear_tweet_layer = nn.Linear(in_features=seq_len, out_features=1).cuda()
sig = sigmoid(linear_tweet_layer(tweet_layer))
sequence_layer = torch.flatten(sig).cuda()
print('output sigmoid per sequence: ', sequence_layer, 'shape: ', sequence_layer.shape)


linear_sequence_layer = nn.Linear(in_features=len(new_input1[0]), out_features=1).cuda()
sig = sigmoid(linear_sequence_layer(sequence_layer))
user_layer = torch.flatten(sig).cuda()
print('output sigmoid for the user: ', user_layer, 'shape: ', user_layer.shape)

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss(reduction='sum').cuda()
fds =  torch.tensor([0.0]).float().cuda()

print(fds, fds.shape)
loss = loss_fn(user_layer.float(), fds)

print('loss: ', loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()
