from transformers import BertModel
import torch
import torch.nn as nn


class DimABSA(nn.Module):
    def __init__(self, hidden_size, bert_model_type, num_category):
        super(DimABSA, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_type)
        hidden_size = self.bert.config.hidden_size
        self.classifier_a_start = nn.Linear(hidden_size, 2)
        self.classifier_a_end = nn.Linear(hidden_size, 2)
        self.classifier_ao_start = nn.Linear(hidden_size, 2)
        self.classifier_ao_end = nn.Linear(hidden_size, 2)
        self.classifier_o_start = nn.Linear(hidden_size, 2)
        self.classifier_o_end = nn.Linear(hidden_size, 2)
        self.classifier_oa_start = nn.Linear(hidden_size, 2)
        self.classifier_oa_end = nn.Linear(hidden_size, 2)
        self.classifier_category = nn.Linear(hidden_size, num_category)
        self.classifier_valence = nn.Linear(hidden_size, 1)
        self.classifier_arousal = nn.Linear(hidden_size, 1)

    def forward(self, query_tensor, query_mask, query_seg, step):
        # ✅ 維度清理：確保是 [B, L]
        while query_tensor.dim() > 2:
            query_tensor = query_tensor.squeeze(1)
        while query_mask.dim() > 2:
            query_mask = query_mask.squeeze(1)
        while query_seg.dim() > 2:
            query_seg = query_seg.squeeze(1)

        # ✅ 長度對齊
        min_len = min(query_tensor.size(1), query_mask.size(1), query_seg.size(1))
        query_tensor = query_tensor[:, :min_len]
        query_mask = query_mask[:, :min_len]
        query_seg = query_seg[:, :min_len]

        # ✅ 防止意外 scalar 或空 tensor
        if query_tensor.dim() != 2:
            query_tensor = query_tensor.view(1, -1)
        if query_mask.dim() != 2:
            query_mask = query_mask.view(1, -1)
        if query_seg.dim() != 2:
            query_seg = query_seg.view(1, -1)

        hidden_states = self.bert(
            query_tensor,
            attention_mask=query_mask,
            token_type_ids=query_seg
        )[0]

        if step == 'A':
            return self.classifier_a_start(hidden_states), self.classifier_a_end(hidden_states)
        elif step == 'O':
            return self.classifier_o_start(hidden_states), self.classifier_o_end(hidden_states)
        elif step == 'AO':
            return self.classifier_ao_start(hidden_states), self.classifier_ao_end(hidden_states)
        elif step == 'OA':
            return self.classifier_oa_start(hidden_states), self.classifier_oa_end(hidden_states)
        elif step == 'C':
            cls_hidden = hidden_states[:, 0, :]
            return self.classifier_category(cls_hidden)
        elif step == 'Valence':
            cls_hidden = hidden_states[:, 0, :]
            return self.classifier_valence(cls_hidden).squeeze(-1)
        elif step == 'Arousal':
            cls_hidden = hidden_states[:, 0, :]
            return self.classifier_arousal(cls_hidden).squeeze(-1)
        else:
            raise KeyError(f"Invalid step: {step}")
# 在 DimABSAModel.py 中定義 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# 在 forward 中使用：
# criterion = FocalLoss(gamma=2)
# loss = criterion(logits, labels)