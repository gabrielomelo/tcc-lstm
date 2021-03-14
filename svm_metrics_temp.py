import pickle
from svm.classifier import SVMClassifier

with open('../fitted_classifiers/svm_classifier_100_stemmed.p', 'rb') as fp:
    svm_classifier = pickle.load(fp)


print(svm_classifier.train_dataset['corpus'].shape)