import numpy as np
import torch
import tqdm

from lstm.LSTMNetwork import LSTMNetwork
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class LSTMHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_scaler(data: np.array, data_range=None):
        """

        :param data: 2D numpy array containing the training data
        :param data_range: data range tuple
        :return:
        """
        if data_range is not None:
            return MinMaxScaler(feature_range=data_range).fit(data)
        return StandardScaler().fit(data)

    @staticmethod
    def prepare_data(data, labels, scaler, train_proportion=0.7):
        """
        Transform the dataset into a series of pytorch tensors, it also scale and divide the data according
        to the training proportion.
        :param labels:
        :param data:
        :param scaler:
        :param train_proportion:
        :return:
        """
        if len(data) != len(labels):
            raise Exception('The dimensions between the data and labels are different, please fix it.')

        train_data, train_labels = [], []
        test_data, test_labels = [], []
        rnn_input = []

        for sequence in tqdm.tqdm(data):
            rnn_input.append(scaler.transform(sequence))

        for i in range(0, len(rnn_input)):
            if i / len(rnn_input) > train_proportion:
                test_data.append(rnn_input[i])
                test_labels.append(labels[i])
            else:
                train_data.append(rnn_input[i])
                train_labels.append(labels[i])

        return np.array(train_data, dtype=np.float32), train_labels, np.array(test_data, dtype=np.float32), test_labels

    @staticmethod
    def train(model: LSTMNetwork, train_data: torch.Tensor, train_labels: torch.Tensor, num_epochs, _lr=0.001):
        """
        train a LSTMNetwork object
        :param _lr:
        :param num_epochs:
        :param model:
        :param train_data:
        :param train_labels:
        :return:
        """
        loss_fn = torch.nn.MSELoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=_lr)
        train_hist = np.zeros(num_epochs)

        for t in tqdm.tqdm(range(0, num_epochs)):
            model.reset_hidden_state()
            prediction = model(train_data)

            loss = loss_fn(prediction.float(), train_labels.float())
            train_hist[t] = loss.item()
            print(f'Epoch {t} train loss: {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model.eval(), train_hist

    @staticmethod
    def create_sequences(corpus: list, labels: list, seq_len: int) -> tuple:
        """

        :param labels:
        :param corpus:
        :param seq_len:
        :return: all data (data and labels) dictionary with
        """
        temp_xs, temp_ys = [], []
        xs, ys = [], []

        for i in range(0, len(corpus)):
            if len(corpus[i]) > seq_len:
                tweet_sequences = []
                for j in range(len(corpus[i]) - seq_len - 1):
                    tweet_sequences.append(
                        corpus[i][j:(j + seq_len)]
                    )
                if tweet_sequences:
                    temp_xs.append(tweet_sequences)
                    temp_ys.append(labels[i])

        for i in range(0, len(temp_xs)):
            for j in range(0, len(temp_xs[i])):
                xs.append(temp_xs[i][j])
                ys.append(temp_ys[i])

        return xs, ys
