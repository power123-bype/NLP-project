import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Initialize model and tokenizer globally to avoid reloading
_MODEL_NAME = 'distilbert-base-uncased'
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading Text Encoder on {_DEVICE}...")
try:
    _TOKENIZER = DistilBertTokenizer.from_pretrained(_MODEL_NAME)
    _MODEL = DistilBertModel.from_pretrained(_MODEL_NAME).to(_DEVICE)
    _MODEL.eval()
except Exception as e:
    print(f"Warning: Failed to load DistilBERT model: {e}")
    _TOKENIZER = None
    _MODEL = None

def get_text_embedding(text, max_length=32):
    """
    Input: text string
    Output: torch.Tensor, shape=(768,), on CPU
    """
    if _MODEL is None:
        raise RuntimeError("Model not initialized")
        
    inputs = _TOKENIZER(
        str(text),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(_DEVICE)
    attention_mask = inputs['attention_mask'].to(_DEVICE)
    
    with torch.no_grad():
        outputs = _MODEL(input_ids=input_ids, attention_mask=attention_mask)
        # Get [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :] # (1, 768)
        
    return cls_embedding.squeeze(0).cpu()
