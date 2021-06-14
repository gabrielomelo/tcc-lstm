import numpy as np
import random

from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from jsonstream import loads
from sklearn.metrics.pairwise import cosine_similarity


with open(r'C:/Datasets/sliced_twitter/stemmed_positive.txt', encoding="utf8") as json_file:
    it = loads(json_file.read())
    document_pos = list(it)
    
with open(r'C:/Datasets/sliced_twitter/stemmed_negative.txt', encoding="utf8") as json_file:
    it = loads(json_file.read())
    document_neg = list(it)


documents = document_pos[0]['user'] + document_neg[0]['user']
data = []
for i in range(len(documents)):
    for j in range(len(documents[i]['tweets'])):
        data.append(documents[i]['tweets'][j]['content'])


random.shuffle(data)
trim_range = 1000000
model_name = 'stemmed_twitter_trimmed_1000000.model'


trimmed_data = []
pos = []
for i in range(trim_range):
    trimmed_data.append(data[i])
    pos.append(i)


tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(trimmed_data)]


model = Doc2Vec(vector_size = 300,
                window_size = 15,
                min_count = 1,
                sampling_threshold = 1e-5,
                negative_size = 5,
                train_epoch = 100, 
                dm = 0, 
                worker_count = 1)
  
model.build_vocab(tagged_data)

for epoch in tqdm(range(100)):
    model.train(tagged_data,
                total_examples = model.corpus_count,
                epochs = model.iter)

model.save(model_name)


model = Doc2Vec.load(model_name)
r = np.random.randint(0, len(trimmed_data), 100)
sum_compar = []
tweet_pos = []
for i in tqdm(range(100)):
    test_data = trimmed_data[r[i]].split()
    tokens = test_data
    new_vector = model.infer_vector(tokens, epochs=10000)
    sims = model.docvecs.most_similar([new_vector], topn=100)
    sum_compar.append(sims)
    tweet_pos.append(r[i])


count_1 = 0
for i in range(len(sum_compar)):
    if sum_compar[i][0][0] == str(tweet_pos[i]):
        count_1 += 1


count_10 = 0
for i in range(len(sum_compar)):
    for i in range(10):
        if sum_compar[i][j][0] == str(tweet_pos[i]):
            count_10 += 1
            break


count_100 = 0
for i in range(len(sum_compar)):
    for i in range(100):
        if sum_compar[i][j][0] == str(tweet_pos[i]):
            count_100 += 1
            break