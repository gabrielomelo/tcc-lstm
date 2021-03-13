from sklearn import svm
import time


class SVMClassifier:
    def __init__(self, train_dataset: dict, test_dataset: dict,
                 _max_iter=-1, _tol=0.001, _shrinking=False, _verbose=True):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.results = None
        self.clf = svm.SVC(shrinking=_shrinking, tol=_tol, max_iter=_max_iter, verbose=_verbose)
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

    def get_results(self):
        return self.results
