# rdrop.py
import torch
import torch.nn.functional as F

def compute_kl_loss(p, q, pad_mask=None):
    """
    計算KL散度損失
    
    Args:
        p: 第一次前向傳播的logits
        q: 第二次前向傳播的logits
        pad_mask: padding mask (可選)
    
    Returns:
        KL散度損失
    """
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss


class RDropLoss:
    """R-Drop regularization"""
    
    def __init__(self, alpha=4.0):
        """
        Args:
            alpha: R-Drop損失的權重係數
        """
        self.alpha = alpha
    
    def compute_loss(self, model, batch_dict, compute_forward_fn, criterion):
        """
        計算R-Drop損失
        
        Args:
            model: 模型
            batch_dict: batch數據
            compute_forward_fn: 計算forward的函數
            criterion: 原始損失函數
        
        Returns:
            總損失 = (loss1 + loss2) / 2 + alpha * kl_loss
        """
        # 第一次前向傳播
        logits1, loss1 = compute_forward_fn(model, batch_dict, criterion)
        
        # 第二次前向傳播（不同的dropout）
        logits2, loss2 = compute_forward_fn(model, batch_dict, criterion)
        
        # 計算KL散度
        kl_loss = compute_kl_loss(logits1, logits2)
        
        # 總損失
        total_loss = (loss1 + loss2) / 2 + self.alpha * kl_loss
        
        return total_loss, kl_loss