import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from svm.classifier import SVMClassifier
from svm.vectorizer import SVMVectorizer

dataset_path = r'../../unvectorized.p'
classifier_path = r'../fitted_classifiers/svm_classifier_100_stemmed.p'
vectorizer_path = r'../fitted_classifiers/svm_vectorizer_100_stemmed.p'

vectorized_dataset_path = r'../lstm_data/converted_vectorized_twitter_100_stemmed.p'
new_classification_dataset_path = '../lstm_data/sentiment_classification_twitter_100_stemmed.p'
new_concat_dataset_path = '../lstm_data/concat_twitter_dataset_100_stemmed.p'
vectorized_dataset = pd.read_pickle(vectorized_dataset_path)


def vectorize_corpus(_vectorizer: SVMVectorizer, _classifier: SVMClassifier, corpus: list):
    _temp = []
    bow = _vectorizer.vectorizer.transform(corpus)
    truncated_bow = _vectorizer.svd.transform(bow)
    scaled_bow = _vectorizer.scaler.transform(truncated_bow)

    classifications = _classifier.clf.predict(scaled_bow)

    for classification in classifications:
        if classification == 0:
            _temp.append(np.array([1.0, 0.0], dtype=np.float32))  # negativo
        elif classification == 4:
            _temp.append(np.array([0.0, 1.0], dtype=np.float32))  # positivo
    return _temp


dataset = pd.read_pickle(dataset_path)
vectorizer = pd.read_pickle(vectorizer_path)
classifier = pd.read_pickle(classifier_path)
classified_dataset = []

for line in tqdm(dataset['user']):
    if len(line['tweets']) > 0:
        temp = []
        for tweet in line['tweets']:
            temp.append(tweet['content'])
        classified_dataset.append(vectorize_corpus(vectorizer, classifier, temp))

with open(new_classification_dataset_path, 'wb+') as fp:
    pickle.dump(classified_dataset, fp)

for i in tqdm(range(0, len(vectorized_dataset['xs']))):
    for j in range(0, len(vectorized_dataset['xs'][i])):
        vectorized_dataset['xs'][i][j] = np.append(vectorized_dataset['xs'][i][j], classified_dataset[i][j])

with open(new_concat_dataset_path, 'wb+') as fp:
    pickle.dump(vectorized_dataset, fp)
