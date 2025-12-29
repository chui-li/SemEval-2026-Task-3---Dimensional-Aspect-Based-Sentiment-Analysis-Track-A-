# category_loss_enhanced.py
# 針對Category分類的增強損失函數

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    """加權的Focal Loss - 根據類別頻率自動調整權重"""
    
    def __init__(self, alpha=None, gamma=2.0, num_classes=None, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # 如果提供了類別數但沒有alpha，自動計算
        if alpha is None and num_classes is not None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) 預測logits
            targets: (N,) 目標標籤
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 動態alpha
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CategoryLossEnhanced(nn.Module):
    """增強的Category損失 - 組合多種策略"""
    
    def __init__(self, num_classes, focal_gamma=2.5, label_smoothing=0.15, 
                 use_class_weights=True):
        super(CategoryLossEnhanced, self).__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.use_class_weights = use_class_weights
        
        # Focal Loss with higher gamma for harder classification
        self.focal_loss = WeightedFocalLoss(
            gamma=focal_gamma, 
            num_classes=num_classes
        )
        
        # Label Smoothing
        self.smooth_loss = LabelSmoothingCrossEntropy(eps=label_smoothing)
    
    def forward(self, inputs, targets):
        """組合Focal Loss和Label Smoothing"""
        focal = self.focal_loss(inputs, targets)
        smooth = self.smooth_loss(inputs, targets)
        
        # 加權組合
        return 0.7 * focal + 0.3 * smooth


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy"""
    
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_preds, target, reduction=self.reduction
        )


def calculate_class_weights(train_data, num_classes):
    """根據訓練數據計算類別權重"""
    class_counts = torch.zeros(num_classes)
    
    # 統計每個類別的樣本數
    for sample in train_data:
        if hasattr(sample, 'category_list'):
            for category in sample.category_list:
                if category < num_classes:
                    class_counts[category] += 1
    
    # 計算權重（使用inverse frequency）
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts + 1e-6)
    
    # 歸一化
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return class_weights