from sklearn import svm
import time


class SVMClassifier:
    def __init__(self, train_dataset: dict, test_dataset: dict, _max_iter=1000000, _tol=0.001, _verbose=True):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.results = None
        self.precision = 0.0
        self.clf = svm.LinearSVC(tol=_tol, max_iter=_max_iter, verbose=_verbose)
        self.train_time = None
        self.classify_time = None

    def train(self):
        start_time = time.time()
        self.clf.fit(self.train_dataset['corpus'], self.train_dataset['y'])
        self.train_time = time.time() - start_time

    def classify(self):
        start_time = time.time()
        self.results = self.clf.predict(self.test_dataset['corpus'])
        self.classify_time = time.time() - start_time
        self.set_precision()

    def set_precision(self):
        true_positive, false_positive = 0, 0
        for i in range(0, len(self.results)):
            if self.results[i] == self.test_dataset['y'][i] and self.results[i] == 4:
                true_positive += 1
            elif self.results[i] != self.test_dataset['y'][i] and self.results[i] == 4:
                false_positive += 1
        self.precision = true_positive / (true_positive + false_positive)

    def get_results(self):
        return self.results
