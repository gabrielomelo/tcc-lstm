import json
import math
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


class SVMVectorizer:
    def __init__(self, file_path: str, database_size=0.5,
                 n_dimensions=25, train_prop=0.7, ds_size_override=None):
        """

        :param file_path:
        :param database_size:
        :param n_dimensions:
        :param train_prop:
        :param ds_size_override:
        """
        self.base_dataset = None
        self.database_size = database_size
        self.train_prop = train_prop
        self.load_data(file_path)
        self.ds_size_override = ds_size_override
        self.corpus_test = {
            'corpus': [],
            'y': []
        }
        self.corpus_training = {
            'corpus': [],
            'y': []
        }
        self.vectorizer = CountVectorizer()
        self.svd = TruncatedSVD(n_dimensions)
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        """

        :param file_path:
        :return:
        """
        with open(file_path, 'r') as fp:
            temp = json.load(fp)
        random.shuffle(temp)
        self.base_dataset = temp[:math.floor(len(temp) * self.database_size)]

    def transform_sentiment(self):
        """

        :return:
        """
        dataset_size = len(self.base_dataset) if self.ds_size_override is None else self.ds_size_override

        for i in range(0, dataset_size):
            if (i + 1) / dataset_size <= self.train_prop:
                self.corpus_training['corpus'].append(self.base_dataset[i]['content'])
                self.corpus_training['y'].append(int(self.base_dataset[i]['sentimentalClassification']))
            else:
                self.corpus_test['corpus'].append(self.base_dataset[i]['content'])
                self.corpus_test['y'].append(int(self.base_dataset[i]['sentimentalClassification']))
        return self

    def vectorize_dataset(self):
        """

        :return:
        """
        self.corpus_training['corpus'] = self.vectorizer.fit_transform(
            self.corpus_training['corpus'],
            self.corpus_training['y']
        )
        self.corpus_test['corpus'] = self.vectorizer.transform(self.corpus_test['corpus'])
        return self

    def scale_dataset(self):
        """

        :return:
        """
        self.corpus_training['corpus'] = self.scaler.fit_transform(
            self.corpus_training['corpus'],
            self.corpus_training['y']
        )
        self.corpus_test['corpus'] = self.scaler.transform(self.corpus_test['corpus'])
        return self

    def truncate(self):
        self.corpus_training['corpus'] = self.svd.fit_transform(
            self.corpus_training['corpus'],
            self.corpus_training['y']
        )
        self.corpus_test['corpus'] = self.svd.transform(self.corpus_test['corpus'])
        return self
