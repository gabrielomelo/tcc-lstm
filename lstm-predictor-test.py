#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch import nn


class CoronaVirusPredictor(nn.Module):

    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(CoronaVirusPredictor, self).__init__()

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)
        self.hidden = None

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )

    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred

    @staticmethod
    def create_sequences(data, _seq_length):
        xs = []
        ys = []

        for i in range(len(data) - _seq_length - 1):
            x = data[i:(i + _seq_length)]
            y = data[i + _seq_length]
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    @staticmethod
    def train_model(_model, train_data, train_labels, test_data=None, test_labels=None, num_epochs=60, _lr=1e-3):
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimiser = torch.optim.Adam(_model.parameters(), lr=_lr)
        _train_hist = np.zeros(num_epochs)
        test_hist = np.zeros(num_epochs)

        for t in range(num_epochs):
            _model.reset_hidden_state()
            y_prediction = _model(train_data)
            loss = loss_fn(y_prediction.float(), train_labels)

            if test_data is not None:
                with torch.no_grad():
                    _y_test_prediction = _model(test_data)
                    test_loss = loss_fn(_y_test_prediction.float(), test_labels)
                test_hist[t] = test_loss.item()

                if t % 10 == 0:
                    print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
            elif t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()}')

            _train_hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return _model.eval(), _train_hist, test_hist


if __name__ == "__main__":
    DAYS_TO_PREDICT = 12
    NUM_EPOCHS = 100
    seq_length = 5

    df = pd.read_csv('../time_series_covid19_confirmed_global.csv')
    df = df.iloc[:, 4:]
    df.isnull().sum().sum()

    daily_cases = df.sum(axis=0)
    daily_cases.index = pd.to_datetime(daily_cases.index)
    daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)

    scaler = StandardScaler()
    scaler = scaler.fit(np.expand_dims(daily_cases, axis=1))
    all_data = scaler.transform(np.expand_dims(daily_cases, axis=1))
    X_all, y_all = CoronaVirusPredictor.create_sequences(all_data, seq_length)
    X_all = torch.from_numpy(X_all).float()
    y_all = torch.from_numpy(y_all).float()

    model = CoronaVirusPredictor(n_features=1, n_hidden=512, seq_len=seq_length, n_layers=2)
    model, train_hist, _ = CoronaVirusPredictor.train_model(model, X_all, y_all, num_epochs=NUM_EPOCHS)

    with torch.no_grad():
        test_seq = X_all[:1]
        predictions = []
        for _ in range(DAYS_TO_PREDICT):
            y_test_prediction = model(test_seq)
            prediction = torch.flatten(y_test_prediction).item()
            predictions.append(prediction)
            new_seq = test_seq.numpy().flatten()
            new_seq = np.append(new_seq, [prediction])
            new_seq = new_seq[1:]
            test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

    predicted_cases = scaler.inverse_transform(
        np.expand_dims(predictions, axis=0)
    ).flatten()

    predicted_index = pd.date_range(
        start=daily_cases.index[-1],
        periods=DAYS_TO_PREDICT + 1,
        closed='right'
    )
    predicted_cases = pd.Series(data=predicted_cases, index=predicted_index)

    plt.plot(predicted_cases, label='Predicted Daily Cases')
    plt.legend()
    plt.show()

    plt.plot(daily_cases, label='Historical Daily Cases')
    plt.plot(predicted_cases, label='Predicted Daily Cases')
    plt.legend()
    plt.show()
