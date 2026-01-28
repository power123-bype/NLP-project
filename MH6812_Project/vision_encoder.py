import os
import torch
from PIL import Image
from torchvision import models, transforms

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

_MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
_MODEL.fc = torch.nn.Identity()  # 输出 fc 前特征：2048 维
_MODEL.eval().to(_DEVICE)

for p in _MODEL.parameters():
    p.requires_grad_(False)

@torch.inference_mode()
def get_image_embedding(path: str) -> torch.Tensor:
    """
    输入：图片路径（可以是 dataset/images/original/... 或 processed/...）
    输出：torch.Tensor, shape=(2048,), 位于 CPU
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    x = _PREPROCESS(img).unsqueeze(0).to(_DEVICE)  # [1,3,224,224]
    feat = _MODEL(x).squeeze(0).detach().cpu()     # [2048]
    return feat

@torch.inference_mode()
def get_image_embedding_batch(paths: list[str], batch_size: int = 32) -> torch.Tensor:
    """
    输入：路径列表
    输出：torch.Tensor, shape=(N,2048), 位于 CPU
    """
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i+batch_size]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image not found: {p}")
            img = Image.open(p).convert("RGB")
            batch.append(_PREPROCESS(img))
        x = torch.stack(batch, dim=0).to(_DEVICE)
        f = _MODEL(x).detach().cpu()
        feats.append(f)
    return torch.cat(feats, dim=0)
