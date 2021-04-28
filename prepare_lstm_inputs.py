import pickle
import itertools
import pandas as pd
from lstm.lstm_helper import LSTMHelper

data_path = '../lstm_experiments/data/converted_vectorized_twitter_100_stemmed.p'
labels_path = '../lstm_experiments/data/converted_labels_twitter_100_stemmed.p'
finished_ds_train_path = '../lstm_experiments/inputs/lstm_train_dataset-textual.p'
finished_ds_test_path = '../lstm_experiments/inputs/lstm_test_dataset-textual.p'
seq_len = 3
dataset_perc = 0.5
train_test_prop = 0.7
classes_prop = (.5, .5)

dataset, labels = pd.read_pickle(data_path), pd.read_pickle(labels_path)
all_data, all_labels = LSTMHelper.create_sequences(dataset, labels, seq_len)

all_data, all_labels = LSTMHelper.balance_dataset(all_data, all_labels, dataset_perc, classes_prop)

scaler = LSTMHelper.get_scaler(list(itertools.chain(*dataset)))

train_data, train_labels, _test_data, _test_labels = LSTMHelper.prepare_data(
    all_data, all_labels, scaler, train_proportion=train_test_prop
)

with open(finished_ds_train_path, 'wb+') as fp:
    pickle.dump((train_data, train_labels), fp)

with open(finished_ds_test_path, 'wb+') as fp:
    pickle.dump((_test_data, _test_labels), fp)
