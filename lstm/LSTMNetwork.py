import torch
import torch.nn as nn


class LSTMNetwork(nn.Module):
    def __init__(self, _input_size: int, _seq_len: int, _hidden_dim: int,
                 _num_layers: int, _dropout=0.5, _dtype=torch.float32):
        """
        LSTMNetwork implementation using pytorch framework
        :param _input_size:
        :param _seq_len:
        :param _hidden_dim:
        :param _num_layers:
        :param _dropout:
        :param _dtype:
        """
        super(LSTMNetwork, self).__init__()
        self.input_size = _input_size
        self.seq_len = _seq_len
        self.hidden_size = _hidden_dim
        self.n_layers = _num_layers
        self.dtype = _dtype
        self.hidden_state = (
            torch.zeros((self.n_layers, self.seq_len, self.hidden_size)).cuda(),
            torch.zeros((self.n_layers, self.seq_len, self.hidden_size)).cuda()
        )
        self.lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=_num_layers
        ).cuda()
        self.linear_hidden = nn.Linear(in_features=self.hidden_size, out_features=1).cuda()
        self.linear_tweet_layer = nn.Linear(in_features=self.seq_len, out_features=1).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward data into the network and return the prediction
        :param data:
        :return:
        """
        lstm_out, self.hidden_state = self.lstm(data, self.hidden_state)
        
        tweet_layer = torch.squeeze(self.linear_hidden(lstm_out)).cuda()
        
        sequence_layer = torch.flatten(self.linear_tweet_layer(tweet_layer)).cuda()
        
        linear_sequence_layer = nn.Linear(in_features=len(data), out_features=1).cuda()

        return self.sigmoid(linear_sequence_layer(sequence_layer))

    def reset_hidden_state(self):
        """
        used to reset the hidden state
        :return:
        """
        self.hidden_state = (
            torch.zeros((self.n_layers, self.seq_len, self.hidden_size)).cuda(),
            torch.zeros((self.n_layers, self.seq_len, self.hidden_size)).cuda()
        )
