import pickle
import itertools
from math import floor
import pandas as pd
from lstm.lstm_helper import LSTMHelper

data_path = '../lstm_experiments/data/converted_vectorized_twitter_100_stemmed.p'
labels_path = '../lstm_experiments/data/converted_labels_twitter_100_stemmed.p'
finished_ds_train_path = '../lstm_experiments/inputs/lstm_train_dataset-textual.p'
finished_ds_test_path = '../lstm_experiments/inputs/lstm_test_dataset-textual.p'
dataset_perc = 0.5
train_prop = 0.7
seq_len = 3

dataset, labels = pd.read_pickle(data_path), pd.read_pickle(labels_path)
if dataset_perc < 1.0:
    dataset, labels = dataset[0: floor(len(dataset)*dataset_perc)], labels[0: floor(len(labels)*dataset_perc)]
all_data, all_labels = LSTMHelper.create_sequences(dataset, labels, seq_len)
scaler = LSTMHelper.get_scaler(list(itertools.chain(*dataset)))

train_data, train_labels, _test_data, _test_labels = LSTMHelper.prepare_data(
    all_data, all_labels, scaler, train_proportion=train_prop
)

with open(finished_ds_train_path, 'wb+') as fp:
    pickle.dump((train_data, train_labels), fp)

with open(finished_ds_test_path, 'wb+') as fp:
    pickle.dump((_test_data, _test_labels), fp)
