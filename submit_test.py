from make_submit import make_submit
import pickle

# 读取测试集预处理
with open('./data_deal/data_test.pkl', 'rb') as f:
    data_test = pickle.load(f)

file_name = 'lstm_3_768_3'
epoch = 15

# 读取测试集预测
with open('./results/%s/new_test_%03d.pkl' % (file_name, epoch), 'rb') as f:
    result_test = pickle.load(f)

f = open('./submit.txt', 'w', encoding='utf-8')
for idx, i in enumerate(data_test):
    submit_one = make_submit(i['text'], result_test[idx])
    f.write(submit_one + '\n')