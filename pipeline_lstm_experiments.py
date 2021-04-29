import os
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from nlp_lib.lstm.lstm_helper import LSTMHelper
from nlp_lib.lstm.depression_detector import DepressionDetector
from nlp_lib.lstm.depression_dataset import DepressionDataset

finished_ds_train_path = '../lstm_experiments/inputs/lstm_train_dataset-textual.p'
finished_ds_test_path = '../lstm_experiments/inputs/lstm_test_dataset-textual.p'
model_save_path = '../lstm_experiments/models/textual/lstm_model_trained_textual_hidden-300_1000-epochs_lr0001_batch-20000_run1'
run_name_logging_dir = '../lstm_experiments/runs/textual/lstm_model_trained_textual_hidden-300_1000-epochs_lr0001_batch-20000_run1'
seq_len = 3
input_size = 300
_hidden_size = 300
n_layers = 4
last_epoch = 0
learning_rate = 0.0001
batch_size = 20000
num_epochs = 1000
_tensorboard_batch = 5
reduction_loss = 'mean'
shuffle = True
just_eval = False
_threshold = None

if not just_eval:
    writer = SummaryWriter(run_name_logging_dir)
    train_data, train_labels = pd.read_pickle(finished_ds_train_path)
    train_dataset = DepressionDataset(train_data, train_labels)

    if os.path.isfile(model_save_path):
        state = torch.load(model_save_path)
        rnn_lstm = DepressionDetector(
            input_size, state['seq_len'], state['hidden_size'], state['n_layers'], state['batch_size']
        ).cuda()
        rnn_lstm.load_state_dict(state['model'])
        optimizer = torch.optim.Adam(rnn_lstm.parameters())
        optimizer.load_state_dict(state['optimizer'])

        model, optimizer, last_epoch = LSTMHelper.train(
            rnn_lstm, optimizer, train_dataset, state['batch_size'], num_epochs, writer,
            state['reduction_loss'], tensorboard_batch=state['tensorboard_batch'],
            _shuffle=state['shuffle'], last_epoch_count=state['last_epoch']
        )
    else:
        rnn_lstm = DepressionDetector(input_size, seq_len, _hidden_size, n_layers, batch_size).cuda()
        optimizer = torch.optim.Adam(rnn_lstm.parameters(), lr=learning_rate)

        model, optimizer, last_epoch = LSTMHelper.train(
            rnn_lstm, optimizer, train_dataset, batch_size, num_epochs, writer,
            reduction_loss, tensorboard_batch=_tensorboard_batch, _shuffle=shuffle
        )

    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'seq_len': seq_len,
            'hidden_size': _hidden_size,
            'n_layers': n_layers,
            'last_epoch': last_epoch,
            'batch_size': batch_size,
            'tensorboard_batch': _tensorboard_batch,
            'reduction_loss': 'mean',
            'shuffle': shuffle,
            'learning_rate': learning_rate
        },
        model_save_path
    )
else:
    state = torch.load(model_save_path)
    rnn_lstm = DepressionDetector(
        input_size, state['seq_len'], state['hidden_size'], state['n_layers'], state['batch_size']
    ).cuda()
    rnn_lstm.load_state_dict(state['model'])

# evaluation of the trained model
test_data, test_labels = pd.read_pickle(finished_ds_test_path)
test_dataset = DepressionDataset(test_data, test_labels)

model, metrics, test_time = LSTMHelper.evaluate(
    rnn_lstm, test_dataset, batch_size, threshold=_threshold
)

print('Tempo classificação (segundos): ', test_time)
print('Precisão: ', metrics.precision)
print('Revocação: ', metrics.recall)
print('Acurácia: ', metrics.accuracy)
print('F1 Score: ', metrics.f1)
print('Threshold: ', metrics.threshold)
