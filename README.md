# 🚀 Enhanced Bidirectional MRC-based ABSA System
This repository provides an enhanced Bidirectional MRC-based model designed for the DimABSA 2025 Task 2 & Task 3 benchmark, supporting Aspect-Opinion extraction and Aspect-Opinion-Category-Valence-Arousal Quadruplet extraction across multiple domains and languages.

# ✨ Key Features
| Enhancement                          | Description                                  |
| ------------------------------------ | -------------------------------------------- |
| **FGM / PGD Adversarial Training**   | 提升模型對對抗樣本的魯棒性                      |
| **EMA (Exponential Moving Average)** | 減少訓練不穩定與過擬合                                  |
| **R-Drop Regularization**            | 增強分類邏輯一致性                                    |
| **Category Loss Enhanced**           | Focal Loss + Label Smoothing + Class Weights |
| **Cosine Warmup Scheduler**          | 更平滑的學習率策略                                    |
| **Multi-GPU Support**                | 自動偵測 GPU 數量並平行訓練                             |
| **Post-processing Optimization**     | Category refinement（語義輔助）提升 Task3 準確度        |

# 📦 Installation
git clone https://github.com/YourRepo/ABSA-Enhanced.git
cd ABSA-Enhanced

# 📂 Directory Structure
.
├─ data/                # Dataset inputs
├─ tasks/        # Inference output results
├─ model/        # Saved checkpoints
├─ log/                 # Training logs
├─ DataProcess.py
├─ DimABSAModel.py
├─ Utils.py
├─ NLP.py
├─ adversarial_training.py
├─ category_loss_enhanced.py
├─ data_augmentation.py
├─ download_model.py
├─ ema.py
├─ ensemble.py
├─ focal_loss.py
├─ postprocess_optimizer.py
├─ pred_zho_restaurant_0.5825.jsonl
├─ rdrop.py
└─ README.md

# ▶️ How to Run
## ⭐ Train

python train.py \
    --task 3 \
    --domain res \
    --language zho \
    --mode train \
    --epoch_num 6 \
    --epoch_num 40 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --tuning_bert_rate 5e-5
    --use_fgm True \
    --adv_epsilon 1.5 \
    --use_ema True \
    --ema_decay 0.9998 \
    --use_focal_loss True \
    --focal_gamma 3.0 \
    --label_smoothing 0.2 \
    --beta 2.0 \
    --scheduler_type cosine \
    --inference_beta 0.9 \

## 🧪 Evaluate

python run_task2_3_trainer_enhanced.py \
    --mode evaluate
    
## 📘 Inference

python run_task2_3_trainer_enhanced.py \
    --mode inference
    
輸出將自動儲存於：
```bash
tasks_reduce/subtask_2/*.jsonl
tasks_reduce/subtask_3/*.jsonl
```

# 📊 Performance

| Epoch | Learning rate | adv_epsilon | label_smoothing | ema_decay | focal_gamma | beta | inference_beta  | drop_alpha | 未輸出數量 | cF1 | 
| ------------- | ------ | ------ | ----------------- | ------------- | ------ | ------ | ----------------- | ----------------- |----------------- |
| 3 | 1e-3 | x | x | x | x | 1 | 0.9 | 4.0 | 16 | 0.5757 | 
| 20 | 2e-5 | 1.0 | 0.2 | 0.999 | 2.5 | 1.5 | 0.82  | 4.0 | 5 | 0.5393 | 
| 30 | 2e-5 | 1.0 | 0.2 | 0.999 | 2.5 | 1.5 | 0.82  | 4.0 | 5 | 0.5561 |  
| 50 | 2e-5 | 1.0 | 0.2 | 0.999 | 2.5 | 1.5 | 0.82  | 4.0 | 5 | 0.5595 | 
| 40 | 2.5e-5 | 1.8 | 0.1 | 0.999 | 2 | 2 | 0.78  | 4.0 | 5 | 0.5471 | 
| 40 | 2e-5 | 1.4 | 0.15 | 0.9995 | 2.5 | 2 | 0.83  | 4.0 | 5 | 0.5587 | 
| 40 | 2e-5 | 1.5 | 0.2 | 0.9998 | 3 | 2 | 0.88  | 4.0 | 6 | 0.5640 | 
| 40 | 5e-5 | 1.5 | 0.2 | 0.9998 | 3 | 2 | 0.9  | 1.0 | 9 | 0.5825 | 





