from sklearn import svm
import time


class SVMClassifier:
    def __init__(self, train_dataset: dict, test_dataset: dict, _max_iter=1000000, _tol=0.001, _verbose=True):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.true_positive, self.false_positive = 0, 0
        self.true_negative, self.false_negative = 0, 0
        self.precision, self.accuracy, self.recall, self.f1 = 0, 0, 0, 0
        self.results, self.train_time, self.classify_time = None, None, None
        self.clf = svm.LinearSVC(tol=_tol, max_iter=_max_iter, verbose=_verbose)

    def train(self):
        start_time = time.time()
        self.clf.fit(self.train_dataset['corpus'], self.train_dataset['y'])
        self.train_time = time.time() - start_time

    def classify(self):
        start_time = time.time()
        self.results = self.clf.predict(self.test_dataset['corpus'])
        self.classify_time = time.time() - start_time
        self.set_variables()
        self.calculate_metrics()

    def set_variables(self):
        for i in range(0, len(self.results)):
            if self.results[i] == self.test_dataset['y'][i] and self.results[i] == 4:
                self.true_positive += 1
            elif self.results[i] != self.test_dataset['y'][i] and self.results[i] == 4:
                self.false_positive += 1
            elif self.results[i] != self.test_dataset['y'][i] and self.results[i] == 0:
                self.false_negative += 1
            elif self.results[i] == self.test_dataset['y'][i] and self.results[i] == 0:
                self.true_negative += 1

    def set_accuracy(self):
        self.accuracy = (self.true_positive + self.true_negative) /\
                        (self.true_positive + self.true_negative + self.false_positive + self.false_negative)
        return self

    def set_recall(self):
        self.recall = self.true_positive / (self.true_positive + self.false_negative)
        return self

    def set_precision(self):
        self.precision = self.true_positive / (self.true_positive + self.false_positive)
        return self

    def set_f1_score(self):
        self.f1 = 2 * ((self.precision * self.recall) / (self.precision + self.recall))
        return self

    def calculate_metrics(self):
        self.set_accuracy().set_recall().set_precision().set_f1_score()

    def get_results(self):
        return self.results
