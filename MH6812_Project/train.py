import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# 导入我们的融合模型
from fusion_model import ImageTextFusionModel

# --- 1. 定义数据集 (Refactored for Offline .pt/.npy Files) ---
class PhotoEditDataset(Dataset):
    def __init__(self, csv_file, img_emb_dir='dataset/cv_embeddings', text_emb_dir='dataset/nlp_embeddings'):
        # Try multiple encodings
        self.raw_data = None
        encodings = ['utf-8', 'gb18030', 'latin1', 'cp1252']
        for enc in encodings:
            try:
                self.raw_data = pd.read_csv(csv_file, engine="python", on_bad_lines="skip", encoding=enc)
                print(f"Successfully read CSV with encoding: {enc}")
                break
            except Exception:
                continue
        
        if self.raw_data is None:
            raise ValueError(f"Could not read {csv_file} with supported encodings.")

        # Ensure absolute paths
        if not os.path.isabs(img_emb_dir):
            img_emb_dir = os.path.join(os.path.dirname(csv_file), 'cv_embeddings')
        if not os.path.isabs(text_emb_dir):
            text_emb_dir = os.path.join(os.path.dirname(csv_file), 'nlp_embeddings')
            
        self.img_emb_dir = os.path.abspath(img_emb_dir)
        self.text_emb_dir = os.path.abspath(text_emb_dir)
        
        print(f"Image Embeddings Dir: {self.img_emb_dir}")
        print(f"Text Embeddings Dir: {self.text_emb_dir}")
        
        # Filter data
        valid_indices = []
        print("Filtering dataset...")
        for idx, row in tqdm(self.raw_data.iterrows(), total=len(self.raw_data)):
            img_filename = os.path.basename(row['original_img_path'])
            img_id = os.path.splitext(img_filename)[0]
            
            # Check Image Embedding
            img_path = os.path.join(self.img_emb_dir, f"{img_id}.pt")
            
            # Check Text Embedding
            text_path = os.path.join(self.text_emb_dir, f"{idx}.pt")
            
            if os.path.exists(img_path) and os.path.exists(text_path):
                valid_indices.append(idx)
        
        self.data = self.raw_data.iloc[valid_indices].reset_index(drop=True)
        self.original_indices = valid_indices
        print(f"Filtered dataset: {len(self.data)} samples (from {len(self.raw_data)})")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        original_idx = self.original_indices[idx]
        
        # 1. 加载 Image Embedding (Task 2 Output)
        # 从原始图片路径中提取 ID，例如 "dataset/images/img_0_orig.jpg" -> "img_0_orig"
        img_filename = os.path.basename(row['original_img_path'])
        img_id = os.path.splitext(img_filename)[0]
        
        # 尝试加载 .pt 或 .npy
        img_emb_path = os.path.join(self.img_emb_dir, f"{img_id}.pt")
        img_npy_path = os.path.join(self.img_emb_dir, f"{img_id}.npy")
        
        img_emb = None
        if os.path.exists(img_npy_path):
             img_emb = torch.from_numpy(np.load(img_npy_path)).float()
        elif os.path.exists(img_emb_path):
             img_emb = torch.load(img_emb_path)
        else:
            raise FileNotFoundError(f"Missing image embedding: {img_id}")

        # 2. 加载 Text Embedding (Task 3 Output)
        # Use original_idx because text embeddings are named after original CSV indices
        text_emb_path = os.path.join(self.text_emb_dir, f"{original_idx}.pt")
        text_npy_path = os.path.join(self.text_emb_dir, f"{original_idx}.npy")
        
        text_emb = None
        if os.path.exists(text_npy_path):
            text_emb = torch.from_numpy(np.load(text_npy_path)).float()
        elif os.path.exists(text_emb_path):
            text_emb = torch.load(text_emb_path)
        else:
             raise FileNotFoundError(f"Missing text embedding: index {original_idx}")
        
        # 3. 获取标签
        labels = torch.tensor([
            row.get('label_brightness', row.get('norm_brightness')),
            row.get('label_contrast', row.get('norm_contrast')),
            row.get('label_temp', row.get('norm_temp'))
        ], dtype=torch.float32)
        
        return img_emb, text_emb, labels

# --- 2. 训练主函数 ---
def train():
    # 配置
    BATCH_SIZE = 64 # 进一步增大 Batch Size 以稳定梯度
    EPOCHS = 50     # 大幅增加 Epochs，配合早停机制
    LR = 0.001
    PATIENCE = 10   # 早停耐心值
    CSV_FILE = '/workspace/MH6812_Project/dataset/metadata.csv'
    
    # 检查数据是否存在
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    # 数据准备
    print("Initializing Dataset with Offline Embeddings...")
    full_dataset = PhotoEditDataset(CSV_FILE)
    
    # 划分训练集和验证集 (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Dataset Split: {train_size} Train, {val_size} Validation")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ImageTextFusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5) # 增加轻微的正则化
    
    # 学习率调度器：当 val_loss 不再下降时，降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # 训练循环
    print(f"Start Training for {EPOCHS} epochs (Early Stopping Patience: {PATIENCE})...")
    
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for img_emb, text_emb, labels in pbar:
            img_emb = img_emb.to(device)
            text_emb = text_emb.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(img_emb, text_emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img_emb, text_emb, labels in val_loader:
                img_emb = img_emb.to(device)
                text_emb = text_emb.to(device)
                labels = labels.to(device)
                
                outputs = model(img_emb, text_emb)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, LR = {current_lr:.6f}")
        
        # Save Best Model & Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"--> New best model saved! (Val Loss: {best_val_loss:.4f})")
            early_stop_counter = 0 # 重置计数器
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter} out of {PATIENCE}")
            
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
        
    print(f"Training Complete. Best Val Loss: {best_val_loss:.4f}")
    # 保存最终模型 (可选)
    torch.save(model.state_dict(), 'model_final.pth')

if __name__ == "__main__":
    train()
