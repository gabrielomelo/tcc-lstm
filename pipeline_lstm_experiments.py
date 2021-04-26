import os
import torch
import pickle
import itertools
import pandas as pd
from math import floor
from torch.utils.tensorboard import SummaryWriter
from lstm.lstm_helper import LSTMHelper
from lstm.depression_detector import DepressionDetector
from lstm.depression_dataset import DepressionDataset

data_path = '../lstm_experiments/data/converted_vectorized_twitter_100_stemmed.p'
labels_path = '../lstm_experiments/data/converted_labels_twitter_100_stemmed.p'
finished_ds_train_path = '../lstm_experiments/inputs/lstm_train_dataset-textual.p'
finished_ds_test_path = '../lstm_experiments/inputs/lstm_test_dataset-textual.p'
model_save_path = '../lstm_experiments/models/textual/lstm_model_trained_textual_hidden-20_100-epochs_lr0001_batch-1000_run1'
run_name_logging_dir = '../lstm_experiments/runs/textual/lstm_model_trained_textual_hidden-20_100-epochs_lr0001_batch-1000_run1'
seq_len = 3
train_prop = 0.7
input_size = 300
_hidden_size = 20
n_layers = 4
learning_rate = 0.0001
batch_size = 1000
num_epochs = 100
_tensorboard_batch = 100
dataset_perc = 0.5
_reduction_loss = 'mean'
shuffle = True
just_eval = False

if not just_eval:

    if not os.path.isfile(finished_ds_train_path):
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

        exit(0)
    else:
        train_data, train_labels = pd.read_pickle(finished_ds_train_path)

    train_dataset = DepressionDataset(train_data, train_labels)

    rnn_lstm = DepressionDetector(input_size, seq_len, _hidden_size, n_layers, batch_size).cuda()
    if os.path.isfile(model_save_path):
        rnn_lstm.load_state_dict(torch.load(model_save_path))
    rnn_lstm.eval()

    writer = SummaryWriter(run_name_logging_dir)

    model = LSTMHelper.train(
        rnn_lstm, train_dataset, batch_size, num_epochs, writer, _lr=learning_rate,
        tensorboard_batch=_tensorboard_batch, reduction_loss=_reduction_loss, _shuffle=shuffle
    )

    torch.save(model.state_dict(), model_save_path)
else:
    rnn_lstm = DepressionDetector(input_size, seq_len, _hidden_size, n_layers, batch_size).cuda()
    if os.path.isfile(model_save_path):
        rnn_lstm.load_state_dict(torch.load(model_save_path))
    rnn_lstm.eval()

# evaluation of the trained model
test_data, test_labels = pd.read_pickle(finished_ds_test_path)

test_dataset = DepressionDataset(test_data, test_labels)
model, metrics, test_time = LSTMHelper.evaluate(rnn_lstm, test_dataset, batch_size, _reduction_loss)

print('Tempo classificação (segundos): ', test_time)
print('Precisão: ', metrics.precision)
print('Revocação: ', metrics.recall)
print('Acurácia: ', metrics.accuracy)
print('F1 Score: ', metrics.f1)
