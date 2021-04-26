import torch
import torch.nn as nn


class DepressionDetector(nn.Module):
    def __init__(self, _input_size: int, _seq_len: int, _hidden_dim: int,
                 _num_layers: int, batch_size: int, _dropout=0.5):
        """
        LSTMNetwork implementation using pytorch framework
        :param _input_size:
        :param _seq_len:
        :param _hidden_dim:
        :param _num_layers:
        :param _dropout:
        """
        super(DepressionDetector, self).__init__()
        self.input_size = _input_size
        self.seq_len = _seq_len
        self.hidden_size = _hidden_dim
        self.n_layers = _num_layers
        self.batch_size = batch_size
        self.hidden_state = self.get_hidden_state()
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=_dropout
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
        out, _ = self.lstm(data, self.hidden_state)
        tweet_layer = torch.squeeze(self.linear_hidden(out))

        return self.sigmoid(
            torch.flatten(
                self.linear_tweet_layer(tweet_layer)
            )
        )

    def get_hidden_state(self):
        """
        used to reset the hidden state
        :return:
        """
        return (
            torch.randn((self.n_layers, self.batch_size, self.hidden_size)).cuda(),
            torch.randn((self.n_layers, self.batch_size, self.hidden_size)).cuda()
        )
