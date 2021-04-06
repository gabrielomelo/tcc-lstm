import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from svm.classifier import SVMClassifier
from svm.vectorizer import SVMVectorizer

dataset_path = r'../lstm_data/unvectorized.p'
classifier_path = r'../fitted_classifiers/svm_classifier_100_stemmed.p'
vectorizer_path = r'../fitted_classifiers/svm_vectorizer_100_stemmed.p'

vectorized_dataset_path = r'../lstm_data/converted_vectorized_twitter_100_stemmed.p'
new_classification_dataset_path = r'../lstm_data/sentiment_classification_twitter_100_stemmed.p'
new_concat_dataset_path = r'../lstm_data/concat_twitter_dataset_100_stemmed.p'
vectorized_dataset = pd.read_pickle(vectorized_dataset_path)


def vectorize_corpus(_vectorizer: SVMVectorizer, _classifier: SVMClassifier, corpus: list):
    _temp, _temp2 = [], []
    bow = _vectorizer.vectorizer.transform(corpus)
    truncated_bow = _vectorizer.svd.transform(bow)
    scaled_bow = _vectorizer.scaler.transform(truncated_bow)

    classifications = _classifier.clf.predict(scaled_bow)

    for classification in classifications:
        if classification == 0:
            _temp.append([0])  # sentimento negativo
            _temp2.append(np.array([0, 1], dtype=np.float32))
        elif classification == 4:
            _temp.append([1])  # sentimento positivo
            _temp2.append(np.array([1, 0], dtype=np.float32))
    return _temp, _temp2


dataset = pd.read_pickle(dataset_path)
vectorizer = pd.read_pickle(vectorizer_path)
classifier = pd.read_pickle(classifier_path)
classified_dataset, classified_dataset2 = [], []

for line in tqdm(dataset['user']):
    if len(line['tweets']) > 0:
        temp = []
        for tweet in line['tweets']:
            temp.append(tweet['content'])
        single_dim, dual_dim = vectorize_corpus(vectorizer, classifier, temp)
        classified_dataset.append(dual_dim)
        classified_dataset2.append(single_dim)

with open(new_classification_dataset_path, 'wb+') as fp:
    pickle.dump(classified_dataset, fp)

for i in tqdm(range(0, len(vectorized_dataset))):
    for j in range(0, len(vectorized_dataset[i])):
        vectorized_dataset[i][j] = np.concatenate(
            (vectorized_dataset[i][j], classified_dataset2[i][j]),
            dtype=np.float32
        )

with open(new_concat_dataset_path, 'wb+') as fp:
    pickle.dump(vectorized_dataset, fp)
