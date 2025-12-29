# ensemble.py
import torch
import numpy as np
from collections import Counter

class ModelEnsemble:
    """模型集成"""
    
    def __init__(self, model_paths, model_class, args, weights=None):
        """
        Args:
            model_paths: 模型權重文件路徑列表
            model_class: 模型類
            args: 參數
            weights: 各模型權重（可選）
        """
        self.models = []
        self.weights = weights if weights else [1.0] * len(model_paths)
        
        # 加載所有模型
        for path in model_paths:
            model = model_class(args.hidden_size, args.bert_model_type, args.category_num)
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['net'])
            if args.gpu:
                model = model.cuda()
            model.eval()
            self.models.append(model)
    
    def predict_with_vote(self, inference_fn, *args, **kwargs):
        """
        使用投票法進行預測
        
        Args:
            inference_fn: 推理函數
            *args, **kwargs: 推理函數的參數
        
        Returns:
            集成後的預測結果
        """
        all_predictions = []
        
        for model in self.models:
            # 對每個模型進行推理
            predictions = inference_fn(model, *args, **kwargs)
            all_predictions.append(predictions)
        
        # 投票
        ensemble_predictions = self._vote_results(all_predictions)
        return ensemble_predictions
    
    def predict_with_average(self, batch_dict, forward_fn):
        """
        使用加權平均進行預測（用於概率輸出）
        
        Args:
            batch_dict: batch數據
            forward_fn: 前向傳播函數
        
        Returns:
            加權平均的預測分數
        """
        weighted_scores = None
        
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                scores = forward_fn(model, batch_dict)
                if weighted_scores is None:
                    weighted_scores = weight * scores
                else:
                    weighted_scores += weight * scores
        
        return weighted_scores / sum(self.weights)
    
    def _vote_results(self, all_predictions):
        """對多個預測結果進行投票"""
        # 這裡需要根據具體的預測格式來實現
        # 以下是一個簡化的示例
        ensemble_pred = []
        
        # 假設predictions是列表的列表
        num_samples = len(all_predictions[0])
        for i in range(num_samples):
            sample_preds = [pred[i] for pred in all_predictions]
            # 這裡可以實現更複雜的投票邏輯
            # 簡單示例：選擇出現最多的預測
            ensemble_pred.append(sample_preds[0])  # 暫時返回第一個模型的預測
        
        return ensemble_pred


class PredictionAggregator:
    """預測結果聚合器"""
    
    @staticmethod
    def aggregate_triplets(triplet_lists, method='union', min_vote=2):
        """
        聚合多個模型的triplet預測
        
        Args:
            triplet_lists: 多個模型預測的triplet列表
            method: 'union' 或 'vote'
            min_vote: 投票法時的最小投票數
        
        Returns:
            聚合後的triplet列表
        """
        if method == 'union':
            # 取並集
            all_triplets = []
            for triplets in triplet_lists:
                all_triplets.extend(triplets)
            # 去重
            unique_triplets = list(set(map(tuple, all_triplets)))
            return [list(t) for t in unique_triplets]
        
        elif method == 'vote':
            # 投票法
            triplet_counter = Counter()
            for triplets in triplet_lists:
                for triplet in triplets:
                    triplet_counter[tuple(triplet)] += 1
            
            # 只保留得票數 >= min_vote 的triplet
            voted_triplets = [list(t) for t, count in triplet_counter.items() if count >= min_vote]
            return voted_triplets
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")