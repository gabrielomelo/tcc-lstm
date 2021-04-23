import os
import pickle

import pandas as pd
import itertools

import torch

from lstm.LSTMHelper import LSTMHelper
from lstm.LSTMNetwork import LSTMNetwork

data_path = '../lstm_data/sentiment_classification_twitter_100_stemmed.p'
labels_path = '../lstm_data/converted_labels_twitter_100_stemmed.p'
finished_ds_train_path = '../lstm_train_dataset.p'
finished_ds_test_path = '../lstm_test_dataset.p'
seq_len = 5
train_prop = 0.7
input_size = 2
_hidden_size = 4
n_layers = 4
learning_rate = 0.001
num_epochs = 50

if not os.path.isfile(finished_ds_train_path):
    dataset = pd.read_pickle(data_path)
    labels = pd.read_pickle(labels_path)

    all_data, all_labels = LSTMHelper.create_sequences(dataset, labels, seq_len)
    scaler = LSTMHelper.get_scaler(list(itertools.chain(*dataset)))

    train_data, train_labels, _test_data, _test_labels = LSTMHelper.prepare_data(
        all_data, all_labels, scaler, train_proportion=train_prop
    )

    with open(finished_ds_train_path, 'wb+') as fp:
        pickle.dump((train_data, train_labels), fp)

    with open(finished_ds_test_path, 'wb+') as fp:
        pickle.dump((_test_data, _test_labels), fp)

    exit(0)
else:
    with open(finished_ds_train_path, 'rb') as fp:
        train_data, train_labels = pickle.load(fp)

rnn_lstm = LSTMNetwork(input_size, seq_len, _hidden_size, n_layers)

model_eval, train_history = LSTMHelper.train(
    rnn_lstm, torch.from_numpy(train_data).cuda(), torch.Tensor(train_labels).cuda(), num_epochs, _lr=learning_rate
)
