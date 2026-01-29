import torch
from transformers import DistilBertTokenizer, DistilBertModel
import os

# 配置
MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 32
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局加载一次模型，避免重复加载
print(f"Loading NLP Model: {MODEL_NAME} on {_DEVICE}...")
try:
    _TOKENIZER = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    _MODEL = DistilBertModel.from_pretrained(MODEL_NAME).to(_DEVICE)
    _MODEL.eval()
except Exception as e:
    print(f"Warning: Failed to load NLP model: {e}")
    _TOKENIZER = None
    _MODEL = None

def get_text_embedding(text: str) -> torch.Tensor:
    """
    输入：文本字符串
    输出：torch.Tensor, shape=(768,), 位于 CPU
    """
    if _MODEL is None or _TOKENIZER is None:
        raise RuntimeError("NLP Model not initialized successfully.")
    
    # 简单的异常处理
    if not text or not isinstance(text, str):
        text = ""

    # Tokenization
    inputs = _TOKENIZER(
        text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(_DEVICE)
    attention_mask = inputs['attention_mask'].to(_DEVICE)
    
    with torch.no_grad():
        outputs = _MODEL(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token (索引 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :] # (1, 768)
        
    return cls_embedding.squeeze(0).cpu()

if __name__ == "__main__":
    # 测试
    test_text = "Make it brighter and warmer."
    emb = get_text_embedding(test_text)
    print(f"Test text: '{test_text}'")
    print(f"Embedding shape: {emb.shape}")
