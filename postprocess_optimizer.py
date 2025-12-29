# postprocess_optimizer.py
# 針對預測結果的後處理優化

import torch
import torch.nn.functional as F

class CategoryPostProcessor:
    """Category預測的後處理"""
    
    def __init__(self, category_dict, confidence_threshold=0.5, 
                 use_second_best=True):
        self.category_dict = category_dict
        self.confidence_threshold = confidence_threshold
        self.use_second_best = use_second_best
    
    def refine_category_prediction(self, category_scores, asp_text, opi_text):
        """
        基於aspect和opinion的文本信息優化category預測
        
        Args:
            category_scores: (num_categories,) category的logits
            asp_text: aspect文本
            opi_text: opinion文本
        
        Returns:
            refined_category: 優化後的category索引
        """
        probs = F.softmax(category_scores, dim=-1)
        top2_probs, top2_indices = torch.topk(probs, k=min(2, len(probs)))
        
        # 如果最高置信度很低，考慮第二高的
        if top2_probs[0] < self.confidence_threshold and len(top2_indices) > 1:
            # 基於文本的啟發式規則
            first_cat = top2_indices[0].item()
            second_cat = top2_indices[1].item()
            
            # 如果第二個類別的概率差距不大，可以考慮切換
            if top2_probs[1] > 0.3 * top2_probs[0]:
                # 這裡可以加入領域知識的規則
                if self._is_more_reasonable(second_cat, asp_text, opi_text):
                    return second_cat
        
        return top2_indices[0].item()
    
    def _is_more_reasonable(self, category_idx, asp_text, opi_text):
        """基於文本判斷category是否更合理"""
        # 簡單的啟發式規則
        category_name = self._get_category_name(category_idx)
        
        if not category_name:
            return False
        
        # 合併文本
        combined_text = asp_text + opi_text
        
        # 餐廳領域的規則
        restaurant_rules = {
            'FOOD': ['菜', '食物', '味道', '口味', '料理', '飯', '麵', '湯', '肉', '菜色', '食材'],
            'DRINKS': ['飲料', '茶', '咖啡', '酒', '水', '果汁', '飲品', '啤酒', '紅酒'],
            'SERVICE': ['服務', '態度', '店員', '服務生', '人員', '接待', '效率', '親切'],
            'AMBIENCE': ['環境', '氛圍', '裝潢', '音樂', '空間', '座位', '燈光', '佈置'],
            'LOCATION': ['位置', '交通', '地點', '停車', '方便', '附近', '距離'],
            'PRICES': ['價格', '價錢', '費用', '便宜', '貴', '划算', '值得', 'CP值'],
            'QUALITY': ['品質', '質量', '新鮮', '衛生', '乾淨'],
        }
        
        # 筆電領域的規則
        laptop_rules = {
            'DISPLAY': ['螢幕', '屏幕', '顯示', '畫質', '解析度', '亮度'],
            'KEYBOARD': ['鍵盤', '按鍵', '打字', '手感'],
            'BATTERY': ['電池', '續航', '充電', '耗電'],
            'PERFORMANCE': ['效能', '性能', '速度', '運行', '處理'],
            'PRICE': ['價格', '價錢', '費用', '便宜', '貴'],
            'DESIGN': ['外觀', '設計', '造型', '質感', '顏色'],
            'PORTABILITY': ['攜帶', '重量', '輕薄', '體積'],
        }
        
        # 飯店領域的規則
        hotel_rules = {
            'ROOMS': ['房間', '客房', '臥室', '床', '床墊'],
            'FACILITIES': ['設施', '設備', '游泳池', '健身房'],
            'SERVICE': ['服務', '人員', '態度', '接待', '櫃檯'],
            'LOCATION': ['位置', '地點', '交通', '景點', '附近'],
            'CLEANLINESS': ['清潔', '乾淨', '衛生', '整潔'],
            'COMFORT': ['舒適', '舒服', '溫度', '安靜'],
            'PRICE': ['價格', '價錢', '費用', '便宜', '貴'],
        }
        
        # 選擇合適的規則集
        rules = restaurant_rules  # 默認使用餐廳規則
        
        # 根據category_dict判斷領域
        if any('LAPTOP' in name or 'DISPLAY' in name for name in self.category_dict.keys()):
            rules = laptop_rules
        elif any('HOTEL' in name or 'ROOMS' in name for name in self.category_dict.keys()):
            rules = hotel_rules
        
        # 檢查關鍵詞匹配
        for key, keywords in rules.items():
            if key in category_name:
                if any(kw in combined_text for kw in keywords):
                    return True
        
        return False
    
    def _get_category_name(self, idx):
        """獲取category名稱"""
        for name, index in self.category_dict.items():
            if index == idx:
                return name
        return None


class TripletRanker:
    """對預測的triplet進行排序和過濾"""
    
    def __init__(self, min_confidence=0.3):
        self.min_confidence = min_confidence
    
    def rank_and_filter(self, triplets_with_scores):
        """
        對triplets按置信度排序並過濾
        
        Args:
            triplets_with_scores: [(triplet, score), ...]
        
        Returns:
            filtered_triplets: 過濾後的triplets
        """
        if not triplets_with_scores:
            return []
        
        # 按分數排序
        sorted_triplets = sorted(
            triplets_with_scores, 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 過濾低置信度
        filtered = [
            triplet for triplet, score in sorted_triplets 
            if score >= self.min_confidence
        ]
        
        # 去重（保留高分的）
        seen = set()
        unique_filtered = []
        for triplet in filtered:
            # 創建唯一標識（使用aspect和opinion位置）
            if len(triplet) >= 4:
                key = tuple(triplet[:4])  # [a_start, a_end, o_start, o_end]
                if key not in seen:
                    seen.add(key)
                    unique_filtered.append(triplet)
        
        return unique_filtered


def optimize_beta_threshold(dev_predictions, dev_targets, beta_range=(0.7, 0.95)):
    """
    在驗證集上搜索最優的beta閾值
    
    Args:
        dev_predictions: 驗證集的預測結果（包含概率）
        dev_targets: 驗證集的標準答案
        beta_range: beta的搜索範圍
    
    Returns:
        best_beta: 最優的beta值
        best_f1: 對應的F1分數
    """
    best_f1 = 0
    best_beta = 0.9
    
    for beta in [x / 100 for x in range(int(beta_range[0] * 100), int(beta_range[1] * 100) + 1, 5)]:
        # 使用當前beta過濾預測
        filtered_preds = [
            pred for pred in dev_predictions 
            if pred.get('confidence', 1.0) >= beta
        ]
        
        # 計算F1
        f1 = calculate_f1(filtered_preds, dev_targets)
        
        if f1 > best_f1:
            best_f1 = f1
            best_beta = beta
    
    return best_beta, best_f1


def calculate_f1(predictions, targets):
    """
    計算F1分數
    
    Args:
        predictions: 預測結果列表
        targets: 標準答案列表
    
    Returns:
        f1: F1分數
    """
    if not predictions or not targets:
        return 0.0
    
    # 將預測和目標轉換為集合（使用前5個元素：aspect位置、opinion位置、category）
    pred_set = set()
    for p in predictions:
        if isinstance(p, dict):
            # 如果是字典格式
            key = (p.get('a_start'), p.get('a_end'), p.get('o_start'), 
                   p.get('o_end'), p.get('category'))
        elif isinstance(p, (list, tuple)) and len(p) >= 5:
            # 如果是列表格式
            key = tuple(p[:5])
        else:
            continue
        pred_set.add(key)
    
    target_set = set()
    for t in targets:
        if isinstance(t, dict):
            key = (t.get('a_start'), t.get('a_end'), t.get('o_start'), 
                   t.get('o_end'), t.get('category'))
        elif isinstance(t, (list, tuple)) and len(t) >= 5:
            key = tuple(t[:5])
        else:
            continue
        target_set.add(key)
    
    # 計算TP, FP, FN
    tp = len(pred_set & target_set)
    fp = len(pred_set - target_set)
    fn = len(target_set - pred_set)
    
    # 計算precision, recall, F1
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return f1


def apply_confidence_threshold_filtering(triplets, min_confidence=0.5):
    """
    基於置信度過濾triplets
    
    Args:
        triplets: [(triplet_info, confidence), ...] 或 [triplet_info with 'confidence' key, ...]
        min_confidence: 最小置信度閾值
    
    Returns:
        filtered_triplets: 過濾後的triplets
    """
    filtered = []
    
    for item in triplets:
        if isinstance(item, tuple) and len(item) == 2:
            # 格式: (triplet, confidence)
            triplet, confidence = item
            if confidence >= min_confidence:
                filtered.append(triplet)
        elif isinstance(item, dict) and 'confidence' in item:
            # 格式: {'triplet': ..., 'confidence': ...}
            if item['confidence'] >= min_confidence:
                filtered.append(item)
        else:
            # 沒有置信度信息，保留
            filtered.append(item)
    
    return filtered


class AdaptiveThresholdOptimizer:
    """自適應閾值優化器"""
    
    def __init__(self, initial_beta=0.9, search_range=(0.7, 0.95), search_step=0.05):
        self.beta = initial_beta
        self.search_range = search_range
        self.search_step = search_step
        self.history = []
    
    def optimize(self, predictions_with_confidence, ground_truth):
        """
        在給定的預測結果上優化beta閾值
        
        Args:
            predictions_with_confidence: [(prediction, confidence), ...]
            ground_truth: 標準答案
        
        Returns:
            optimal_beta: 最優閾值
        """
        best_f1 = 0
        best_beta = self.beta
        
        beta = self.search_range[0]
        while beta <= self.search_range[1]:
            # 使用當前beta過濾
            filtered = [pred for pred, conf in predictions_with_confidence if conf >= beta]
            
            # 計算F1
            f1 = calculate_f1(filtered, ground_truth)
            
            # 記錄
            self.history.append({'beta': beta, 'f1': f1})
            
            if f1 > best_f1:
                best_f1 = f1
                best_beta = beta
            
            beta += self.search_step
        
        self.beta = best_beta
        return best_beta, best_f1
    
    def get_optimization_curve(self):
        """獲取優化曲線數據"""
        return self.history