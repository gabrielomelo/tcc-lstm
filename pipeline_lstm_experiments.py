import os
import torch
import pickle
import itertools
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from lstm.lstm_helper import LSTMHelper
from lstm.depression_detector import DepressionDetector
from lstm.depression_dataset import DepressionDataset

data_path = '../lstm_experiments/data/sentiment_classification_twitter_100_stemmed.p'
labels_path = '../lstm_experiments/data/converted_labels_twitter_100_stemmed.p'
finished_ds_train_path = '../lstm_experiments/inputs/lstm_train_dataset-sentiment.p'
finished_ds_test_path = '../lstm_experiments/inputs/lstm_test_dataset-sentiment.p'
model_save_path = '../lstm_experiments/models/sentiment/lstm_model_trained_sentiment_adam-mean_100-epochs_hidden-6_lr0001_batch-100000_run6'
run_name_logging_dir = '../lstm_experiments/runs/sentiment/lstm_model_trained_sentiment_adam-mean_100-epochs_hidden-6_lr0001_batch-100000_run6'
seq_len = 5
train_prop = 0.7
input_size = 2
_hidden_size = 6
n_layers = 4
learning_rate = 0.0001
batch_size = 100000
num_epochs = 100
_tensorboard_batch = 10
_reduction_loss = 'sum'
shuffle = True
just_eval = True

if not just_eval:

    if not os.path.isfile(finished_ds_train_path):
        dataset, labels = pd.read_pickle(data_path), pd.read_pickle(labels_path)
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
