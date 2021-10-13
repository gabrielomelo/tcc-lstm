import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import json
from random import shuffle
import pickle
import pandas as pd

path = 'C:/Users/Zuzu/Desktop/tcc/doc2vec/newModel/stemmed_twitter_trimmed_3192403.model'
model = Doc2Vec.load(path)

datasetPathNegative = 'C:/Users/Zuzu/Desktop/tcc/Data/stemming/steemedData/stemmed_twitter/stemmed_negative.txt'
datasetPathPositive = 'C:/Users/Zuzu/Desktop/tcc/Data/stemming/steemedData/stemmed_twitter/stemmed_positive.txt'
dataPos = None
with open(datasetPathNegative, 'r') as fp:
    data = json.load(fp)

with open(datasetPathPositive, 'r') as fp:
    dataPos = json.load(fp)

for user in dataPos['user']:
    data['user'].append(user)

shuffle(data['user'])

print('começou a vetorização')
for user in data['user']:
    for i in range(0, len(user['tweets'])):
        user['tweets'][i]['content'] = model.infer_vector(user['tweets'][i]['content'].split(), epochs=100)

with open('vectorized_data_final_teste.p', 'wb+') as of:
    pickle.dump(data, of)

print('terminou')
