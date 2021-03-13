import pickle
from svm.pre_process import SVMPreProcess
from svm.classifier import SVMClassifier

dataset_path = 'C:/Users/Zuzu/Desktop/tcc/Data/stemming/steemedData/stemmed_sentiments/stemmed_sentiments.txt'
database_size = 0.5
n_dimensions = 25
train_prop = 0.7

pre_process = SVMPreProcess(dataset_path, database_size, n_dimensions, train_prop)

pre_process.transform_sentiment().vectorize_dataset().truncate()

svm_classifier = SVMClassifier(pre_process.corpus_training, pre_process.corpus_test)

print('Iniciou o treinamento, dimensões: ', pre_process.corpus_training['corpus'].shape())
svm_classifier.train()

print('Acabou o treinamento, iniciou a classificação', pre_process.corpus_training['corpus'].shape())
svm_classifier.classify()

print('Acabou a classificação')
with open('svm_classifier_25.p', 'wb') as handle:
    pickle.dump(svm_classifier, handle)