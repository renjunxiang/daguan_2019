import pickle
import os
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from net import MyDataset, collate_fn, dataset
from net import Net
from net.dataset import text2bert
import pandas as pd
import argparse
from pytorch_transformers import BertTokenizer, BertModel

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', default='0', help='cuda:0/1/2')
parser.add_argument('--pretrain', default='bert', help='texts_all_160_24_1_1_100000')
parser.add_argument('--num_layers', default=3, type=int, help='lstm layernum 3/4')
parser.add_argument('--hidden_dim', default=768, type=int, help='lstm hidden 768/1024')
parser.add_argument('--loss_weight', default=1, type=int, help='loss:2/3')
parser.add_argument('--epochs', default=25, type=int, help='epochs')
parser.add_argument('--k', default=0.865, type=float, help='k')
parser.add_argument('--size', default=768, type=int, help='size')
opt = parser.parse_args()

dataset.device = "cuda:%s" % opt.cuda
device = dataset.device
# torch.manual_seed(1)


embedding_dim = opt.size
num_layers = opt.num_layers
hidden_dim = opt.hidden_dim
BS = 64
epochs = opt.epochs

# 导入文本编码、词典
with open('./data_deal/word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

# 读取训练集预处理
with open('./data_deal/data_train.pkl', 'rb') as f:
    data_train = pickle.load(f)

# 读取测试集预处理
with open('./data_deal/data_test.pkl', 'rb') as f:
    data_test = pickle.load(f)

# 拆分训练集
data_train1, data_train2 = train_test_split(data_train,
                                            test_size=0.1,
                                            random_state=1)
trainloader1 = torch.utils.data.DataLoader(
    dataset=MyDataset(data_train1),
    batch_size=BS, shuffle=True, collate_fn=collate_fn)

k = opt.k
for loss_weight in [3]:
    for bert_name in [
        opt.pretrain
    ]:
        bert_path = './bert/' + bert_name + '/'
        dataset.tokenizer = BertTokenizer.from_pretrained(bert_path)
        dataset.BERT = BertModel.from_pretrained(bert_path).to(device)
        dataset.BERT.eval()
        F1_ = 0
        while F1_ < 0.5:
            # vocab_size还有pad和unknow，要+2
            model = Net(embedding_dim,
                        num_layers,
                        hidden_dim,
                        device=device).to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.001)

            file_name = 'bert_%s_lstm_%d_%d_%d' % (
                bert_name, num_layers, hidden_dim, loss_weight)

            if not os.path.exists('./results/%s/' % (file_name)):
                os.mkdir('./results/%s/' % (file_name))

            score = []
            for epoch in range(epochs):
                print('Start Epoch: %d\n' % (epoch + 1))
                sum_loss = 0.0
                model.train()
                for i, data in enumerate(trainloader1):
                    sentence_features, mask_loss, entities_start, entities_end = data

                    # 训练ner
                    model.zero_grad()

                    # ner损失
                    loss = model.cal_loss(sentence_features,
                                          mask_loss,
                                          entities_start,
                                          entities_end,
                                          [1] + [loss_weight] * 3)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item()

                    if (i + 1) % 50 == 0:
                        print('\nEpoch: %d ,batch: %d' % (epoch + 1, i + 1))
                        print('ner_loss: %f' % (sum_loss / 50))
                        sum_loss = 0.0



                # train2得分=====================================================================
                model.eval()
                p_len = 0.001
                l_len = 0.001
                correct_len = 0.001
                score_list = []
                entity_list_all = []

                for idx, data in enumerate(data_train2):
                    model.zero_grad()
                    text = data['text']
                    with torch.no_grad():
                        entity_predict = model(text)

                    entity_list_all.append(entity_predict)

                    p_set = set(entity_predict)
                    p_len += len(p_set)
                    l_set = set(data['labels'])
                    l_len += len(l_set)
                    correct_len += len(p_set.intersection(l_set))

                    if (idx + 1) % 2000 == 0:
                        print('finish train_2 %d' % (idx + 1))

                Precision = correct_len / p_len
                Recall = correct_len / l_len
                F1 = 2 * Precision * Recall / (Precision + Recall)

                score.append([epoch + 1,
                              round(Precision, 4), round(Recall, 4), round(F1, 4)])
                print('\nEpoch: %d ,Precision:%f, Recall:%f, F1:%f' % (epoch + 1, Precision, Recall, F1))

                score1_df = pd.DataFrame(score,
                                         columns=['Epoch',
                                                  'P', 'R', 'F1'])
                print(score1_df)
                score1_df.to_csv('./results/%s/new_train_2.csv' %
                                 (file_name), index=False)
                F1_ = max(F1_, F1)
                if F1 >= k:
                    # 保存网络参数
                    torch.save(model.state_dict(),
                               './results/%s/new_param_%03d.pth' %
                               (file_name, epoch + 1))
                    torch.save(model, './results/%s/new_%03d.pth' %
                               (file_name, epoch + 1))

                    with open('./results/%s/new_train_2_%03d.pkl' %
                              (file_name, epoch + 1), 'wb') as f:
                        pickle.dump(entity_list_all, f)

                    # eval预测结果=====================================================================

                    model.eval()
                    entity_list_all = []

                    for idx, data in enumerate(data_test):
                        model.zero_grad()
                        text = data['text']
                        with torch.no_grad():
                            entity_predict = model(text)

                        entity_list_all.append(entity_predict)

                        if (idx + 1) % 1000 == 0:
                            print('finish dev %d' % (idx + 1))
                    with open('./results/%s/new_test_%03d.pkl' %
                              (file_name, epoch + 1), 'wb') as f:
                        pickle.dump(entity_list_all, f)
