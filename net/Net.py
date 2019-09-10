import torch
import torch.nn as nn
from .module import Features
from .task import Locate_Entity
from .dataset import text2bert


class Net(nn.Module):
    def __init__(self,
                 embedding_dim=300,
                 num_layers=3,
                 hidden_dim=256,
                 device='cpu'):
        super(Net, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim

        # 文本信息
        self.get_features = Features(num_layers,
                                     embedding_dim,
                                     hidden_dim,
                                     device)

        # entity起止位置
        self.get_entity_score = Locate_Entity(hidden_dim,
                                              device)

    def cal_loss(self,
                 text_features,
                 mask_loss,
                 entity_B_labels,
                 entity_E_labels,
                 loss_weight):

        # 计算实体文本语义
        text_features, mask_idx = self.get_features(text_features, mask_loss)

        # 预测entity的起止
        entity_B_scores, entity_E_scores = self.get_entity_score(text_features)

        # 计算entity的损失,去除mask部分
        loss_weight = torch.Tensor(loss_weight).to(self.device)
        loss = self.get_entity_score.cal_loss(entity_B_scores,
                                              entity_E_scores,
                                              entity_B_labels,
                                              entity_E_labels,
                                              mask_idx,
                                              loss_weight)

        return loss

    def forward(self, text):
        # 词嵌入结果
        text_features, mask = text2bert([text])

        # 计算实体文本语义
        text_features, _ = self.get_features(text_features, mask)

        # 预测实体
        entity_predict = []
        entity_B_scores, entity_E_scores = self.get_entity_score(text_features)
        entity_B_scores = entity_B_scores[0].argmax(-1).tolist()
        entity_E_scores = entity_E_scores[0].argmax(-1).tolist()

        for entity_B_idx, entity_B_score in enumerate(entity_B_scores):
            if entity_B_score > 0:
                # E是在B之后的,索引从B开始
                for entity_E_idx, entity_E_score in enumerate(entity_E_scores[entity_B_idx:]):
                    if entity_E_score == entity_B_score:
                        entity_predict.append((entity_B_idx, entity_B_idx + entity_E_idx, entity_B_score))
                        break

        return entity_predict
