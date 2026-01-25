import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm

# 导入我们的模型和模拟接口
from fusion_model import ImageTextFusionModel
from vision_encoder import get_image_embedding
from text_encoder import get_text_embedding

# --- 1. 定义数据集 ---
class PhotoEditDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 获取输入
        img_path = row['original_img_path']
        instruction = row['instruction']
        
        # 使用 CV 和 NLP 组员提供的接口获取特征
        # 注意：在实际训练中，为了速度，通常会先预计算好 Embedding 存下来，
        # 而不是在训练循环里每次都跑 ResNet/BERT。
        # 这里为了演示流程，我们实时调用。
        img_emb = get_image_embedding(img_path)
        text_emb = get_text_embedding(instruction)
        
        # 获取标签 (假设已归一化)
        labels = torch.tensor([
            row['label_brightness'],
            row['label_contrast'],
            row['label_temp']
        ], dtype=torch.float32)
        
        return img_emb, text_emb, labels

# --- 2. 训练主函数 ---
def train():
    # 配置
    BATCH_SIZE = 2
    EPOCHS = 5
    LR = 0.001
    CSV_FILE = 'metadata.csv'
    
    # 检查数据是否存在
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please run create_mock_data.py first.")
        return

    # 数据准备
    dataset = PhotoEditDataset(CSV_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ImageTextFusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # 训练循环
    print("Start Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for img_emb, text_emb, labels in pbar:
            img_emb = img_emb.to(device)
            text_emb = text_emb.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = model(img_emb, text_emb)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
    # 保存模型
    torch.save(model.state_dict(), 'model_final.pth')
    print("Model saved to model_final.pth")

if __name__ == "__main__":
    train()
