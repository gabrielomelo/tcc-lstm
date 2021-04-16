import numpy as np
import torch
from lstm.LSTMNetwork import LSTMNetwork
from sklearn.preprocessing import MinMaxScaler


class LSTMHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_scaler(data: np.array, labels: np.array, data_range=(0, 1)):
        """

        :param data: 2D numpy array containing the training data
        :param labels: 2D numpy array containing the training data labels
        :param data_range: data range tuple
        :return:
        """
        scaler = MinMaxScaler(feature_range=data_range)
        scaler.fit(data, labels)
        return scaler

    @staticmethod
    def train(model: LSTMNetwork, train_data, train_labels, test_data=None, test_labels=None):
        """
        train a LSTMNetwork object
        :param model:
        :param train_data:
        :param train_labels:
        :param test_data:
        :param test_labels:
        :return:
        """
        loss_fn = torch.nn.CrossEntropy(Lossreduction='sum')

        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        num_epochs = 60

        train_hist = np.zeros(num_epochs)
        test_hist = np.zeros(num_epochs)

        for t in range(num_epochs):
            model.reset_hidden_state()

        y_prediction = model(train_data)

        loss = loss_fn(y_prediction.float(), train_labels)

        if test_data is not None:
            with torch.no_grad():
                y_test_prediction = model(test_data)
                test_loss = loss_fn(y_test_prediction.float(), test_labels)
                test_hist[t] = test_loss.item()

        if t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')

        train_hist[t] = loss.item()

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

        return model.eval(), train_hist, test_hist
