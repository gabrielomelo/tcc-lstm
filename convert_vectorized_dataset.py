import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

dataset_path = '../../vectorized.p'
converted_dataset_path = '../lstm_data/converted_vectorized_twitter_100_stemmed.p'
dataset = pd.read_pickle(dataset_path)

xs, ys = [], []

for line in tqdm(dataset['user']):
    if len(line['tweets']) > 0:
        temp = []
        for tweet in line['tweets']:
            temp.append(tweet['content'])
        xs.append(temp)
        ys.append(float(line['hasDepression']))

converted_dataset = {'xs': np.array(xs, dtype=list), 'ys': np.array(ys, dtype=list)}

print('gravando essa porra')
with open(converted_dataset_path, 'wb+') as fp:
    pickle.dump(converted_dataset, fp)
