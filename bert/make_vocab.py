import pickle

# 导入文本编码
with open('./data_deal/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

special = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']

f = open('./data_deal/vocab.txt', 'w', encoding='utf-8')
for i in special:
    f.write(i + '\n')

for i, j in tokenizer.index_word.items():
    f.write(j + '\n')
f.close()
