import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTextFusionModel(nn.Module):
    def __init__(self, img_dim=2048, text_dim=768, hidden_dim=512, output_dim=3):
        super(ImageTextFusionModel, self).__init__()
        
        # V3 架构 (Ultimate Version)：残差门控注意力 + 动态缩放融合
        
        # 1. 投影层 (Projection)
        # 使用 SELayer (Squeeze-and-Excitation) 思想增强特征
        self.img_projector = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # BatchNorm -> LayerNorm (更适合小 Batch)
            nn.GELU(), # ReLU -> GELU (更平滑的激活)
            nn.Dropout(0.3)
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 2. 增强型门控网络 (Enhanced Gating)
        # 引入残差连接的思想：不仅用 Gate 筛选，还保留原始信息
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(), # Sigmoid -> Tanh (允许负向抑制)
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 3. 融合层
        self.fusion_dim = hidden_dim * 2
        
        # 4. 回归头 (ResNet-style MLP)
        # 加深网络并引入残差连接
        self.fc1 = nn.Linear(self.fusion_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 保持维度以便做残差
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(0.3)
        
        self.head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, img_emb, text_emb):
        # 0. 维度调整
        if img_emb.dim() == 1: img_emb = img_emb.unsqueeze(0)
        if text_emb.dim() == 1: text_emb = text_emb.unsqueeze(0)

        # 1. 投影
        img_feat = self.img_projector(img_emb)
        text_feat = self.text_projector(text_emb)
        
        # 2. 计算门控 (Gate)
        # 使用文本特征去生成针对图像的注意力图
        gate = self.gate_net(text_feat)
        
        # 3. 应用门控 (Gated Image Features)
        # 关键 Trick: 显式缩放图像特征，防止其范数过大淹没文本
        # 我们将图像特征限制在一定范围内
        img_feat_scaled = img_feat * gate
        
        # 4. 融合 (Concatenation)
        # Trick: 动态文本加权 (Dynamic Reweighting)
        # 不再硬编码 x3.0，而是给文本特征一个更高的基准权重 x5.0，让模型自己去学衰减
        text_feat_boosted = text_feat * 5.0 
        combined = torch.cat((img_feat_scaled, text_feat_boosted), dim=1)
        
        # 5. 回归 (Residual MLP)
        x = self.fc1(combined)
        x = self.act1(x)
        x = self.drop1(x)
        
        # Residual Block
        residual = x
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = x + residual # Skip Connection
        
        output = self.head(x)
        
        return output
