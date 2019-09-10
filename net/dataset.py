import torch
from torch.utils.data import Dataset
from random import choice

max_len = 160
device = "cuda:0"

tokenizer = None
BERT = None


def make_batch(seqs):
    """
    补齐一个batch的seqs
    :param seqs:
    :return:
    """
    texts_len = [min(max_len, len(i)) for i in seqs]
    len_max = min(max_len, max(texts_len))
    seq_batch = [i[:max_len] + [0] * (len_max - texts_len[idx]) for idx, i in enumerate(seqs)]

    return seq_batch


def text2bert(texts):
    """
    进入BERT要补上[CLS]和[SEP]，长度+2
    输出要去掉[CLS]和[SEP]，仅保留 文本+[PAD]
    :param texts:
    :return:
    """
    mask_loss = []
    text_seqs = []
    segments_ids = []

    text_len = [min(max_len, len(text)) for text in texts]
    text_max = max(text_len)

    for num, text in enumerate(texts):
        text_cat = ['[CLS]'] + list(text[:max_len]) + ['[SEP]']
        text_bert = []
        for c in text_cat:
            if c in tokenizer.vocab:
                text_bert.append(c)
            else:
                text_bert.append('[UNK]')

        # 用于损失的mask，除了sentence其余都是0
        mask_loss.append([0] + [1] * (len(text_cat) - 2) + [0] * (text_max - text_len[num] + 1))

        # 输入bert
        text_seq = tokenizer.convert_tokens_to_ids(text_bert) + [0] * (text_max - text_len[num])
        text_seqs.append(text_seq)
        segments_ids.append([0] * (text_max + 2))
    text_seqs = torch.LongTensor(text_seqs).to(device)
    segments_ids = torch.LongTensor(segments_ids).to(device)

    # bert的mask编码
    mask_bert = 1 - torch.eq(text_seqs, 0)
    with torch.no_grad():
        sentence_features, _ = BERT(text_seqs, segments_ids, mask_bert)
    # sentence_features = sentence_features[-1] #新版不需要

    mask_loss = torch.LongTensor(mask_loss).to(device)
    mask_feature = mask_loss.unsqueeze(-1).repeat(1, 1, 768)

    # 最终只保留sentence的序列输出
    sentence_features = torch.where(torch.eq(mask_feature, 0),
                                    torch.zeros_like(sentence_features),
                                    sentence_features)

    return sentence_features[:, 1:-1, :], mask_loss[:, 1:-1]


# 定义数据读取方式
class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data_one = self.dataset[index]

        return data_one

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    texts = [i['text'] for i in batch]
    entities_start = [i['entity_start'] for i in batch]
    entities_end = [i['entity_end'] for i in batch]

    sentence_features, mask_loss = text2bert(texts)
    entities_start = make_batch(entities_start)
    entities_end = make_batch(entities_end)

    entities_start = torch.LongTensor(entities_start).to(device)
    entities_end = torch.LongTensor(entities_end).to(device)

    return sentence_features, mask_loss, entities_start, entities_end
