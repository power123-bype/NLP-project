import torch
import numpy as np

# 模拟 CV Specialist 的工作
def get_image_embedding(image_path):
    """
    Mock function to simulate ResNet50 embedding extraction.
    Returns a random tensor of shape (2048,)
    """
    # 模拟加载图片、预处理、推理的过程
    # print(f"[Mock CV] Processing image: {image_path}")
    
    # 返回一个随机的 2048 维特征向量
    # 模拟 ResNet50 output
    return torch.randn(2048)
