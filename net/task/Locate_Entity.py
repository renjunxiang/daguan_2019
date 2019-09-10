import torch
import torch.nn as nn


class Locate_Entity(nn.Module):
    """
    输入 sentence，预测 entity的首尾位置
    """

    def __init__(self,
                 hidden_dim=256,
                 device='cpu'):
        super(Locate_Entity, self).__init__()

        self.predict_B = nn.Sequential(
            nn.Linear(hidden_dim, 4)
        )
        self.predict_E = nn.Sequential(
            nn.Linear(hidden_dim, 4)
        )
        self.device = device

    def cal_loss(self,
                 s_B_idxs,
                 s_E_idxs,
                 s_B_labels,
                 s_E_labels,
                 mask_idx,
                 weight):
        # mask_idx = (1 - mask_idx).float()
        mask_idx = mask_idx.float()

        # 计算subject_B的损失,提高正样本权重,去除mask部分
        loss1 = nn.CrossEntropyLoss(reduce=False, weight=weight)(s_B_idxs.transpose(2, 1), s_B_labels)
        loss1 = (loss1 * mask_idx).sum() / mask_idx.sum()

        # 计算subject_E的损失,提高正样本权重,去除mask部分
        loss2 = nn.CrossEntropyLoss(reduce=False, weight=weight)(s_E_idxs.transpose(2, 1), s_E_labels)
        loss2 = (loss2 * mask_idx).sum() / mask_idx.sum()

        return loss1 + loss2

    def forward(self, sentence_features):
        s_B_scores = self.predict_B(sentence_features)
        s_E_scores = self.predict_E(sentence_features)

        return s_B_scores, s_E_scores
