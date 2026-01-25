import torch

# 模拟 NLP Specialist 的工作
def get_text_embedding(text):
    """
    Mock function to simulate BERT embedding extraction.
    Returns a random tensor of shape (768,)
    """
    # 模拟 Tokenization, BERT 推理
    # print(f"[Mock NLP] Processing text: {text}")
    
    # 返回一个随机的 768 维特征向量
    # 模拟 BERT [CLS] token output
    return torch.randn(768)
