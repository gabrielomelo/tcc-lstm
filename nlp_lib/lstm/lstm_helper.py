import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
from random import shuffle
from math import floor

from nlp_lib.lstm.depression_detector import DepressionDetector
from nlp_lib.lstm.detector_metrics import DetectorMetrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class LSTMHelper:

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
    def train(
            model: DepressionDetector, optimizer: torch.optim.Adam,
            train_dataset: Dataset, batch_size, num_epochs, writer,
            reduction_loss, tensorboard_batch=100, _shuffle=True, last_epoch_count=0
    ):
        """
        train a LSTMNetwork object
        :param optimizer:
        :param last_epoch_count:
        :param _shuffle:
        :param reduction_loss:
        :param tensorboard_batch:
        :param writer:
        :param batch_size:
        :param train_dataset:
        :param num_epochs:
        :param model:
        :return:
        """
        loss_fn = torch.nn.MSELoss(reduction=reduction_loss).cuda()
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=_shuffle)

        model.train()
        running_loss_epoch = 0.0
        running_loss_batches = 0.0
        last_epoch = last_epoch_count + 1 if last_epoch_count != 0 else last_epoch_count

        for epoch in range(0, num_epochs):
            try:
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
                        print(f'Epoch {epoch + 1} batch {batch + 1} train loss: {loss.item()}')

                        if batch % tensorboard_batch == tensorboard_batch - 1:
                            writer.add_scalar(f'training loss per {tensorboard_batch} batches',
                                              running_loss_batches / tensorboard_batch,
                                              last_epoch * len(train_data_loader) + batch)
                            running_loss_batches = 0.0

                writer.add_scalar('training loss per epoch', running_loss_epoch, last_epoch)
                running_loss_epoch = 0.0
                last_epoch += 1
            except KeyboardInterrupt:
                return model, optimizer, last_epoch
        return model, optimizer, last_epoch

    @staticmethod
    def evaluate(model: DepressionDetector, test_dataset: Dataset, batch_size: int, threshold=None) -> tuple:
        """
        evaluate the trained model using f1, accuracy, precision and recall metrics.
        :param threshold:
        :param model:
        :param test_dataset:
        :param batch_size:
        :return:
        """
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        start_time = time.time()
        predictions, test_labels = [], []

        for batch, data in enumerate(test_data_loader, 0):
            inputs, labels = data
            if inputs.shape[0] == batch_size:
                inputs = inputs.cuda()
                with torch.no_grad():
                    predictions.extend(model(inputs).tolist())
                    test_labels.extend(labels.tolist())
                print(f'Batch {batch + 1} with size {batch_size}')

        return model, \
               DetectorMetrics(predictions, test_labels, threshold=threshold), \
               (time.time() - start_time)

    @staticmethod
    def balance_dataset(_sequences, _labels, _dataset_perc, _classes_prop) -> tuple:
        """

        :param _classes_prop:
        :param _sequences:
        :param _labels:
        :param _dataset_perc:
        :return:
        """
        if len(_labels) != len(_sequences):
            raise Exception("dataset with different size")

        ones, zeros, temp_all, temp_data, temp_labels = [], [], [], [], []
        for i in range(0, len(_labels)):
            if _labels[i] == 1:
                ones.append((_sequences[i], _labels[i]))
            else:
                zeros.append((_sequences[i], _labels[i]))

        dataset_real_size = floor(len(_labels) * _dataset_perc)
        one_minimal_count = floor(dataset_real_size * _classes_prop[0])
        zero_minimal_count = floor(dataset_real_size * _classes_prop[1])

        if one_minimal_count <= len(ones) and zero_minimal_count <= len(zeros):
            shuffle(ones), shuffle(zeros)
            temp_all.extend(ones[0: one_minimal_count])
            temp_all.extend(zeros[0: zero_minimal_count])
            shuffle(temp_all)
        else:
            lower_count = 0
            if one_minimal_count >= len(ones):
                if zero_minimal_count >= len(zeros):
                    lower_count = len(ones) if len(ones) < len(zeros) else len(zeros)
                else:
                    lower_count = len(ones)
            elif zero_minimal_count >= len(zeros):
                if one_minimal_count >= len(ones):
                    lower_count = len(ones) if len(ones) < len(zeros) else len(zeros)
                else:
                    lower_count = len(zeros)
            shuffle(ones), shuffle(zeros)
            temp_all.extend(ones[0: lower_count])
            temp_all.extend(zeros[0: lower_count])
            shuffle(temp_all)

        for t in temp_all:
            temp_data.append(t[0]),
            temp_labels.append(t[1])

        return temp_data, temp_labels

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
