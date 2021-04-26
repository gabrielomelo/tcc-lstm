
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
        self.f1 = 0.0
        self.positive = positive
        self.negative = negative
        if threshold is None:
            self.set_threshold()
        else:
            self.threshold = threshold
        self.set_variables()

    def threshold_fn(self, value: float) -> bool:
        """

        :param value:
        :return:
        """
        return value > self.threshold

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

    def set_threshold(self):
        acc, occur = 0.0, 0
        for j in range(0, len(self.prediction)):
            if self.target[j] == 1.0:
                acc += self.prediction[j]
                occur += 1

        self.threshold = acc / occur

    def eval(self):
        """

        :return:
        """
        return self.set_accuracy().set_recall().set_precision().set_f1_score()

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

    def set_f1_score(self):
        """

        :return:
        """
        try:
            self.f1 = 2 * ((self.precision * self.recall) / (self.precision + self.recall))
        except ZeroDivisionError:
            self.f1 = 0.0
        return self
