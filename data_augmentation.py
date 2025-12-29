# data_augmentation.py
import random

class TextAugmentor:
    """
    文本數據增強
    注意：這個是簡化版，實際使用可能需要安裝 jieba, synonyms 等套件
    如果沒有這些套件，可以暫時跳過數據增強
    """
    
    def __init__(self, aug_prob=0.3):
        self.aug_prob = aug_prob
        self.use_jieba = False
        self.use_synonyms = False
        
        # 嘗試導入套件
        try:
            import jieba
            self.jieba = jieba
            self.use_jieba = True
        except ImportError:
            print("Warning: jieba not installed, synonym replacement disabled")
        
        try:
            import synonyms
            self.synonyms = synonyms
            self.use_synonyms = True
        except ImportError:
            print("Warning: synonyms not installed, synonym replacement disabled")
    
    def augment(self, text, aspect_indices=None, opinion_indices=None):
        """
        增強文本，但保護aspect和opinion的位置不變
        
        Args:
            text: 原始文本
            aspect_indices: aspect的起始和結束索引列表 [(start, end), ...]
            opinion_indices: opinion的起始和結束索引列表 [(start, end), ...]
        
        Returns:
            增強後的文本
        """
        if random.random() > self.aug_prob:
            return text
        
        # 如果沒有安裝必要套件，返回原文
        if not self.use_jieba or not self.use_synonyms:
            return text
        
        # 簡單的增強策略：對非關鍵詞進行同義詞替換
        words = list(self.jieba.cut(text))
        
        # 標記受保護的詞
        protected_positions = set()
        if aspect_indices:
            for start, end in aspect_indices:
                protected_positions.update(range(start, end + 1))
        if opinion_indices:
            for start, end in opinion_indices:
                protected_positions.update(range(start, end + 1))
        
        # 對未受保護的詞進行替換
        new_words = []
        current_pos = 0
        for word in words:
            word_len = len(word)
            if current_pos not in protected_positions and len(word) > 1:
                # 嘗試替換
                synonyms_list = self.synonyms.nearby(word)[0]
                if len(synonyms_list) > 0 and random.random() < 0.3:
                    word = random.choice(synonyms_list[:3])
            new_words.append(word)
            current_pos += word_len
        
        return ''.join(new_words)
    
    def random_swap(self, text, n=2):
        """隨機交換n對詞（簡單版本）"""
        if not self.use_jieba:
            return text
        
        words = list(self.jieba.cut(text))
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ''.join(words)