from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型名稱
model_name = "hfl/chinese-roberta-wwm-ext-large"

# 指定下載路徑 (你想存到哪裡)
local_dir = "/mnt/usr1/azure005/NLP/models/chinese-roberta-wwm-ext-large"

# 下載 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
tokenizer.save_pretrained(local_dir)

# 下載模型
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=False)
model.save_pretrained(local_dir)

print(f"✅ 模型下載完成，存放在: {local_dir}")
