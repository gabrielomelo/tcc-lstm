import os
import torch
import pickle
import itertools
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from lstm.lstm_helper import LSTMHelper
from lstm.depression_detector import DepressionDetector
from lstm.depression_dataset import DepressionDataset

data_path = '../lstm_data/sentiment_classification_twitter_100_stemmed.p'
labels_path = '../lstm_data/converted_labels_twitter_100_stemmed.p'
finished_ds_train_path = '../lstm_train_dataset.p'
finished_ds_test_path = '../lstm_test_dataset.p'
model_save_path = '../lstm_model_trained'
run_name_logging_dir = 'runs/sentiment_classification_run_lstm_2'
seq_len = 5
train_prop = 0.7
input_size = 2
_hidden_size = 500
n_layers = 4
learning_rate = 0.0001
batch_size = 1000
num_epochs = 100

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

train_dataset = DepressionDataset(train_data, train_labels)

rnn_lstm = DepressionDetector(input_size, seq_len, _hidden_size, n_layers, batch_size).cuda()
if os.path.isfile(model_save_path):
    rnn_lstm.load_state_dict(torch.load(model_save_path))
rnn_lstm.eval()

writer = SummaryWriter(run_name_logging_dir)

model = LSTMHelper.train(
    rnn_lstm, train_dataset, batch_size, num_epochs, writer, _lr=learning_rate
)

torch.save(model.state_dict(), model_save_path)

print('done')
