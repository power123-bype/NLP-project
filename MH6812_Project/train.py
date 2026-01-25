import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm

# 导入我们的融合模型
from fusion_model import ImageTextFusionModel

# 注意：我们不再需要导入 vision_encoder 和 text_encoder
# 因为数据已经由 Task 2 和 Task 3 预处理好并保存为 .pt 文件了

# --- 1. 定义数据集 (Refactored for Offline .pt Files) ---
class PhotoEditDataset(Dataset):
    def __init__(self, csv_file, img_emb_dir='dataset/embeddings/images', text_emb_dir='dataset/embeddings/texts'):
        self.data = pd.read_csv(csv_file)
        self.img_emb_dir = img_emb_dir
        self.text_emb_dir = text_emb_dir
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. 加载 Image Embedding (Task 2 Output)
        # 从原始图片路径中提取 ID，例如 "dataset/images/img_0_orig.jpg" -> "img_0_orig"
        img_filename = os.path.basename(row['original_img_path'])
        img_id = os.path.splitext(img_filename)[0]
        img_emb_path = os.path.join(self.img_emb_dir, f"{img_id}.pt")
        
        # 2. 加载 Text Embedding (Task 3 Output)
        # 直接使用行号 idx
        text_emb_path = os.path.join(self.text_emb_dir, f"{idx}.pt")
        
        try:
            img_emb = torch.load(img_emb_path)
            text_emb = torch.load(text_emb_path)
        except FileNotFoundError as e:
            # 实际开发中应该记录日志或跳过，这里直接报错以便调试
            raise FileNotFoundError(f"Missing embedding file: {e.filename}. Make sure Task 2/3 have generated all .pt files.")
        
        # 3. 获取标签
        labels = torch.tensor([
            row['label_brightness'],
            row['label_contrast'],
            row['label_temp']
        ], dtype=torch.float32)
        
        return img_emb, text_emb, labels

# --- 2. 训练主函数 ---
def train():
    # 配置
    BATCH_SIZE = 4 # 增大 Batch Size 因为现在读取速度很快
    EPOCHS = 5
    LR = 0.001
    CSV_FILE = 'metadata.csv'
    
    # 检查数据是否存在
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please run create_mock_data.py first.")
        return

    # 数据准备
    print("Initializing Dataset with Offline Embeddings...")
    try:
        dataset = PhotoEditDataset(CSV_FILE)
        # 测试读取一个样本，确保路径没问题
        _ = dataset[0]
    except Exception as e:
        print(f"Dataset initialization failed: {e}")
        print("Tip: Run 'generate_offline_features.py' to create mock .pt files first.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"DataLoader ready. Batch Size: {BATCH_SIZE}, Total Samples: {len(dataset)}")
    
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
            # 关键：将 CPU Tensor 移动到 GPU
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
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
    # 保存模型
    torch.save(model.state_dict(), 'model_final.pth')
    print("Model saved to model_final.pth")

if __name__ == "__main__":
    train()
