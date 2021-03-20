import pickle
from svm.vectorizer import SVMVectorizer
from svm.classifier import SVMClassifier


output_path = r'..\svm_classifier_300_stemmed.p'
dataset_path = r'C:\Users\Gabriel\Desktop\tcc-lstm\v2\pre-process-sentiment\sliced_sentiment\sliced_stemmed_sentiments.txt'
database_size = 0.25
n_dimensions = 300
train_prop = 0.7
tolerance = 0.001
_ds_size_override = None

pre_process = SVMVectorizer(dataset_path, database_size, n_dimensions, train_prop, ds_size_override=_ds_size_override)

pre_process.transform_sentiment().vectorize_dataset().truncate().scale_dataset()

svm_classifier = SVMClassifier(pre_process.corpus_training, pre_process.corpus_test, _tol=tolerance)

print('Iniciou o treinamento, dimensões: ', pre_process.corpus_training['corpus'].shape)
svm_classifier.train()

print('Acabou o treinamento, iniciou a classificação', pre_process.corpus_test['corpus'].shape)
svm_classifier.classify()

print('Tempo treino: ', svm_classifier.train_time)
print('Tempo classificação: ', svm_classifier.classify_time)
print('Precisão: ', svm_classifier.precision)
print('Revocação: ', svm_classifier.recall)
print('Acurácia: ', svm_classifier.accuracy)
print('F1 Score: ', svm_classifier.f1)

print('Salvando o modelo.')
with open(output_path, 'wb+') as fp:
    pickle.dump(svm_classifier, fp)
