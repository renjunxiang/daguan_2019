from gensim.models import word2vec
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', default=300, type=int, help='Dimensionality of the word vectors')
parser.add_argument('--sg', default=1, type=int, help='1 for skip-gram; otherwise CBOW')

opt = parser.parse_args()
size = opt.size
sg = opt.sg

model = word2vec.Word2Vec.load("./data_deal/word2vec/size_%d_sg_%d.model" % (size, sg))

# 导入文本编码、词典
with open('./data_deal/word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

index_word = {index: word for word, index in word_index.items()}

weight = [[0] * size]
for index, word in index_word.items():
    if word in model:
        weight.append(model[word])
    else:
        weight.append([0.001] * size)
weight.append([0.001] * size)
weight = np.array(weight)
print(weight.shape)

# 保存权重
with open("./data_deal/weight/size_%d_sg_%d.pkl" % (size, sg), 'wb') as f:
    pickle.dump(weight, f)
# nohup python make_weight.py --size 300 --sg 1 2>&1 >word2vec &
