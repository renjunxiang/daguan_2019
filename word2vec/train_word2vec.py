from gensim.models import word2vec
import pickle
import logging
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

with open('./data_deal/texts_all.pkl', 'rb') as f:
    texts_all = pickle.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--size', default=300, type=int, help='Dimensionality of the word vectors')
parser.add_argument('--sg', default=0, type=int, help='1 for skip-gram; otherwise CBOW')
parser.add_argument('--workers', default=4, type=int)

opt = parser.parse_args()
size = opt.size
sg = opt.sg
workers = opt.workers

model = word2vec.Word2Vec(texts_all,
                          size=size,
                          sg=sg,
                          workers=workers,
                          min_count=2,
                          sorted_vocab=1)

# 保存模型，供日後使用
model.save("./data_deal/word2vec/size_%d_sg_%d.model" % (size, sg))
# nohup python train_word2vec.py --size 300 --sg 1 --workers 6 2>&1 >word2vec &
