from math import sqrt
from sklearn import metrics
from numpy import argmax
from matplotlib import pyplot


class DetectorMetrics:
    def __init__(self, prediction: list, target: list, positive=1.0, negative=0.0, threshold=None):
        self.prediction = prediction
        self.target = target
        self.true_positive = 0.0
        self.false_positive = 0.0
        self.true_negative = 0.0
        self.false_negative = 0.0
        self.accuracy = 0.0
        self.recall = 0.0
        self.precision = 0.0
        self.f_score = 0.0
        self.positive = positive
        self.negative = negative
        self.threshold = threshold

    def optimize_g_mean(self):
        """
        optimize the threshold to maximise the g-mean score (ROC Curve)
        :return:
        """
        g_means = []
        fpr, tpr, thresholds = metrics.roc_curve(self.target, self.prediction, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        for i in range(len(fpr)):
            g_means.append(sqrt(tpr[i] * (1 - fpr[i])))
        pyplot.figure()
        idx = argmax(g_means)
        lw = 2
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[idx], g_means[idx]))
        pyplot.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area ={0:.2f})'.format(roc_auc))
        pyplot.scatter(fpr[idx], tpr[idx], marker='o', color='black', label='Best')
        pyplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        pyplot.xlim([0.0, 1.0])
        pyplot.ylim([0.0, 1.05])
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title('ROC Curve')
        pyplot.legend(loc="lower right")
        pyplot.show()
        self.threshold = thresholds[idx]
        self.set_variables()
        self.eval()
        return self

    def threshold_fn(self, value: float) -> bool:
        """

        :param value:
        :return:
        """
        return value >= self.threshold

    def set_variables(self):
        """

        :return:
        """
        for j in range(0, len(self.prediction)):
            if self.threshold_fn(self.prediction[j]) and self.target[j] == self.positive:
                self.true_positive += 1
            elif self.threshold_fn(self.prediction[j]) and self.target[j] == self.negative:
                self.false_positive += 1
            elif not self.threshold_fn(self.prediction[j]) and self.target[j] == self.negative:
                self.true_negative += 1
            elif not self.threshold_fn(self.prediction[j]) and self.target[j] == self.positive:
                self.false_negative += 1
        return self

    def eval(self):
        """

        :return:
        """
        return self.set_accuracy().set_recall().set_precision().set_f_score()

    def set_accuracy(self):
        """

        :return:
        """
        try:
            self.accuracy = (self.true_positive + self.true_negative) /\
                            (self.true_positive + self.true_negative + self.false_positive + self.false_negative)
        except ZeroDivisionError:
            self.accuracy = 0.0
        return self

    def set_recall(self):
        """

        :return:
        """
        try:
            self.recall = self.true_positive / (self.true_positive + self.false_negative)
        except ZeroDivisionError:
            self.recall = 0
        return self

    def set_precision(self):
        """

        :return:
        """
        try:
            self.precision = self.true_positive / (self.true_positive + self.false_positive)
        except ZeroDivisionError:
            self.precision = 0.0
        return self

    def set_f_score(self):
        """

        :return:
        """
        try:
            self.f_score = (2 * self.precision * self.recall) / (self.precision + self.recall)
        except ZeroDivisionError:
            self.f_score = 0.0
        return self
