import pandas as pd
import itertools
import os
import pickle
from lstm.LSTMHelper import LSTMHelper
from lstm.LSTMNetwork import LSTMNetwork

data_path = '../lstm_data/sentiment_classification_twitter_100_stemmed.p'
labels_path = '../lstm_data/converted_labels_twitter_100_stemmed.p'
finished_ds_path = '../lstm_data/converted_labels_twitter_100_stemmed.p'

seq_len = 5
train_prop = 0.7
input_size = 2
_hidden_size = 4
n_layers = 4
_learning_rate = 0.001
num_epochs = 50

dataset = pd.read_pickle(data_path)
labels = pd.read_pickle(labels_path)

all_data, all_labels = LSTMHelper.create_sequences(dataset, labels, seq_len)
scaler = LSTMHelper.get_scaler(list(itertools.chain(*dataset)))

train_data, train_labels, _test_data, _test_labels = LSTMHelper.prepare_data(
    all_data, all_labels, scaler, train_proportion=train_prop
)

rnn_lstm = LSTMNetwork(input_size, seq_len, _hidden_size, n_layers)

model_eval, train_history, test_history = LSTMHelper.train(
    rnn_lstm, train_data, train_labels, num_epochs,
    test_data=_test_data, test_labels=_test_labels, learning_rate=_learning_rate
)
