import codecs
import os
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer

# 全部文本
texts_all = []

f_text_all = open('./data_deal/texts_all.txt', 'w', encoding='utf-8')

# 导入训练集
f = codecs.open('./data/train.txt', 'r', encoding='utf-8')
for line in f:
    line = line.replace('\n', '')
    line_list = line.split('  ')
    text = []

    for entity_raw in line_list:
        # 获取文本
        text_seq = entity_raw[:-2].split('_')
        text.extend(text_seq)
    texts_all.append(text)
    f_text_all.write(' '.join(text) + '\n')
f.close()
print('finish train')

# 导入测试集
f = codecs.open('./data/test.txt', 'r', encoding='utf-8')
for line in f:
    line = line.replace('\n', '')

    # 获取文本
    text = line.split('_')
    texts_all.append(text)
    f_text_all.write(' '.join(text) + '\n')
f.close()
print('finish test')

# 导入语料集
f = codecs.open('./data/corpus.txt', 'r', encoding='utf-8')
for idx, line in enumerate(f):
    line = line.replace('\n', '')

    # 获取文本
    text = line.split('_')
    texts_all.append(text)
    f_text_all.write(' '.join(text) + '\n')
    if (idx + 1) % 100000 == 0:
        print('finish corpus %d' % (idx + 1))
f.close()

with open('./data_deal/texts_all.pkl', 'wb') as f:
    pickle.dump(texts_all, f)

f_text_all.close()

# 构建词典
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_all)

print('words_num:', len(tokenizer.word_index))  # 7542

for i in tokenizer.index_word:
    if tokenizer.word_counts[tokenizer.index_word[i]] == 1:
        words_num = i - 1
        break
else:
    words_num = len(tokenizer.index_word)
print('words_num,count>1:', words_num)

with open('./data_deal/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

word_index = tokenizer.word_index
word_index = {i: j for i, j in word_index.items() if j <= words_num}
with open('./data_deal/word_index.pkl', 'wb') as f:
    pickle.dump(word_index, f)

for i in tokenizer.index_word:
    if tokenizer.word_counts[tokenizer.index_word[i]] == 1:
        words_num = i - 1
        break
else:
    words_num = len(tokenizer.index_word)
print('words_num,count>1:', words_num)

with open('./data_deal/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

word_index = tokenizer.word_index
word_index = {i: j for i, j in word_index.items() if j <= words_num}
with open('./data_deal/word_index.pkl', 'wb') as f:
    pickle.dump(word_index, f)
