import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm

from lstm.depression_detector import DepressionDetector
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class LSTMHelper:
    BATCH_QTY_TENSORBOARD = 100

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
    def train(model: DepressionDetector, train_dataset: Dataset, batch_size, num_epochs, writer, _lr=0.001):
        """
        train a LSTMNetwork object
        :param writer:
        :param batch_size:
        :param train_dataset:
        :param _lr:
        :param num_epochs:
        :param model:
        :return:
        """
        loss_fn = torch.nn.MSELoss(reduction='mean').cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=_lr)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        model.train()
        running_loss_epoch = 0.0
        running_loss_batches = 0.0
        for epoch in range(0, num_epochs):
            for batch, data in enumerate(train_data_loader, 0):
                inputs, labels = data
                if inputs.shape[0] == batch_size:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    out = model(inputs)
                    loss = loss_fn(out.float(), labels.float())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss_batches += loss.item()
                    running_loss_epoch += loss.item()
                    print(f'Epoch {epoch+1} batch {batch + 1} train loss: {loss.item()}')

                    if batch % LSTMHelper.BATCH_QTY_TENSORBOARD == LSTMHelper.BATCH_QTY_TENSORBOARD - 1:
                        writer.add_scalar(f'training loss per {LSTMHelper.BATCH_QTY_TENSORBOARD} batches',
                                          running_loss_batches / LSTMHelper.BATCH_QTY_TENSORBOARD,
                                          epoch * len(train_data_loader) + batch)
                        running_loss_batches = 0.0

            writer.add_scalar('training loss per epoch',
                              running_loss_epoch,
                              epoch)
            running_loss_epoch = 0.0

        return model.eval()

    @staticmethod
    def evaluate():
        with torch.no_grad():
            pass
        pass

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
