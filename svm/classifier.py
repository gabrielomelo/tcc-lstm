from sklearn import svm


class SVMClassifier:
    def __init__(self, train_dataset: dict, test_dataset: dict,
                 _max_iter=-1, _tol=0.001, _shrinking=False, _verbose=True):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.results = None
        self.clf = svm.SVC(shrinking=_shrinking, tol=_tol, max_iter=_max_iter, verbose=_verbose)

    def train(self):
        self.clf.fit(self.train_dataset['corpus'], self.train_dataset['y'])

    def classify(self):
        self.results = self.clf.predict(self.test_dataset['corpus'])

    def get_results(self):
        return self.results
