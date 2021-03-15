import pickle
from svm.classifier import SVMClassifier

with open('../fitted_classifiers/svm_classifier_200_stemmed.p', 'rb') as fp:
    svm_classifier = pickle.load(fp)

results = svm_classifier.get_results()
test_dataset = svm_classifier.test_dataset
true_positive, false_positive = 0, 0
true_negative, false_negative = 0, 0

for i in range(0, len(results)):
    if results[i] == test_dataset['y'][i] and results[i] == 4:
        true_positive += 1
    elif results[i] != test_dataset['y'][i] and results[i] == 4:
        false_positive += 1
    elif results[i] != test_dataset['y'][i] and results[i] == 0:
        false_negative += 1
    elif results[i] == test_dataset['y'][i] and results[i] == 0:
        true_negative += 1

accuracy = (true_positive + true_negative) /\
                (true_positive + true_negative + false_positive + false_negative)

recall = true_positive / (true_positive + false_negative)

precision = true_positive / (true_positive + false_positive)

f1 = 2 * ((precision * recall) / (precision + recall))

print('precisao: ', precision)
print('acuracia: ', accuracy)
print('revocacao: ', recall)
print('f1: ', f1)
