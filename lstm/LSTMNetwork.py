import torch
import torch.nn as nn
import numpy as np


class LSTMNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 _dropout=0.5, output_dim=1, _dtype=torch.float32):
        """
        LSTMNetwork implementation using pytorch framework
        :param input_dim:
        :param hidden_dim:
        :param num_layers:
        :param _dropout:
        :param output_dim:
        :param _dtype:
        """
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=_dropout
        )
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self._dtype = _dtype
        self.hidden_state = (
            torch.zeros((self.n_layers, self.seq_len, self.n_hidden), dtype=self._dtype),
            torch.zeros((self.n_layers, self.seq_len, self.n_hidden), dtype=self._dtype)
        )
        self.cell_state = (
            torch.zeros((self.n_layers, self.seq_len, self.n_hidden), dtype=self._dtype),
            torch.zeros((self.n_layers, self.seq_len, self.n_hidden), dtype=self._dtype)
        )
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, data: np.array):
        """
        forward data into the network and return the prediction
        :return:
        """
        lstm_out, self.hidden_state = self.lstm(
            data.view(len(data), self.seq_len, -1),
            (self.hidden_state, self.cell_state)
        )
        last_time_step = lstm_out.view(self.seq_len, len(data), self.n_hidden)[-1]

        return self.linear(last_time_step)

    def reset_hidden_state(self):
        """
        used to reset the hidden state
        :return:
        """
        self.hidden_state = (
            torch.zeros((self.n_layers, self.seq_len, self.n_hidden), dtype=self._dtype),
            torch.zeros((self.n_layers, self.seq_len, self.n_hidden), dtype=self._dtype)
        )
