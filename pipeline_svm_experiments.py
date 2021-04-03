import pickle
from svm.vectorizer import SVMVectorizer
from svm.classifier import SVMClassifier


output_path = r'..\svm_classifier_100_stemmed.p'
output_vect_path = r'..\svm_vectorizer_100_stemmed.p'
dataset_path = r'C:\Users\Gabriel\Desktop\tcc-lstm\datasets\pre-process-sentiment\sliced_sentiment\sliced_stemmed_sentiments.txt'
database_size = 0.25
n_dimensions = 100
train_prop = 0.7
tolerance = 0.001
_ds_size_override = None

vectorizer = SVMVectorizer(dataset_path, database_size, n_dimensions, train_prop, ds_size_override=_ds_size_override)

vectorizer.transform_sentiment().vectorize_dataset().truncate().scale_dataset()

svm_classifier = SVMClassifier(vectorizer.corpus_training, vectorizer.corpus_test, _tol=tolerance)

print('Iniciou o treinamento, dimensões: ', vectorizer.corpus_training['corpus'].shape)
svm_classifier.train()

print('Acabou o treinamento, iniciou a classificação', vectorizer.corpus_test['corpus'].shape)
svm_classifier.classify()

print('Tempo treino: ', svm_classifier.train_time)
print('Tempo classificação: ', svm_classifier.classify_time)
print('Precisão: ', svm_classifier.precision)
print('Revocação: ', svm_classifier.recall)
print('Acurácia: ', svm_classifier.accuracy)
print('F1 Score: ', svm_classifier.f1)

print('Salvando o vetorizador')
with open(output_vect_path, 'wb+') as fp:
    pickle.dump(vectorizer, fp)

print('Salvando o modelo.')
with open(output_path, 'wb+') as fp:
    pickle.dump(svm_classifier, fp)
