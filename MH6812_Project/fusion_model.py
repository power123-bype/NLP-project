import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTextFusionModel(nn.Module):
    def __init__(self, img_dim=2048, text_dim=768, hidden_dim=512, output_dim=3):
        super(ImageTextFusionModel, self).__init__()
        
        # 定义简单的特征融合层
        # 方式 1: 直接拼接 (Concatenation)
        # 输入维度 = img_dim + text_dim
        self.input_dim = img_dim + text_dim
        
        # MLP 回归头 (3个全连接层)
        self.regressor = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, output_dim)
            # 注意：如果 Label 是归一化到 0-1 的，这里可以用 Sigmoid
            # 如果是原始数值（如色温可能很大），则不需要 Sigmoid，直接输出线性值
            # 假设 Label 已归一化，我们暂时不加 Sigmoid，让模型自己学习范围，或者在 Loss 计算前处理
        )
        
    def forward(self, img_emb, text_emb):
        """
        Args:
            img_emb: (batch_size, img_dim) - 视觉特征
            text_emb: (batch_size, text_dim) - 文本特征
        Returns:
            output: (batch_size, output_dim) - 预测的参数 (brightness, contrast, temp)
        """
        # 1. 特征归一化与增强
        # 归一化以平衡不同模态的尺度
        img_emb = F.normalize(img_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        
        # 增强文本特征的权重 (例如提升 2 倍)
        text_emb = text_emb * 2.0
        
        # 确保两个特征都在同一个设备上，且维度正确
        if img_emb.dim() == 1:
            img_emb = img_emb.unsqueeze(0)
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
            
        combined_features = torch.cat((img_emb, text_emb), dim=1)
        
        # 2. MLP 回归
        output = self.regressor(combined_features)
        
        return output
