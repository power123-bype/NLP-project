import os
import torch
import pandas as pd
import numpy as np

def generate_offline_features():
    print("Generating mock offline features (Task 2 & Task 3 simulation)...")
    
    # 1. Check/Load Metadata
    csv_file = 'metadata.csv'
    if not os.path.exists(csv_file):
        print("metadata.csv not found. Please run create_mock_data.py first or ensure data exists.")
        return

    df = pd.read_csv(csv_file)
    print(f"Loaded metadata.csv with {len(df)} rows.")

    # 2. Setup Directories
    img_emb_dir = 'dataset/embeddings/images'
    text_emb_dir = 'dataset/embeddings/texts'
    os.makedirs(img_emb_dir, exist_ok=True)
    os.makedirs(text_emb_dir, exist_ok=True)

    # 3. Simulate Task 2: Generate Image Embeddings (Unique Images)
    # Extract unique image IDs from paths
    # Assuming path format: .../filename.jpg -> ID: filename
    unique_img_paths = df['original_img_path'].unique()
    
    print(f"Simulating Task 2: Processing {len(unique_img_paths)} unique images...")
    
    for img_path in unique_img_paths:
        # Extract ID: "dataset/images/img_0_orig.jpg" -> "img_0_orig"
        img_filename = os.path.basename(img_path)
        img_id = os.path.splitext(img_filename)[0]
        
        save_path = os.path.join(img_emb_dir, f"{img_id}.pt")
        
        # Mock ResNet50 vector (2048,)
        # Ensure it's on CPU and float32
        tensor = torch.randn(2048, dtype=torch.float32)
        torch.save(tensor, save_path)
        
    print(f"Task 2 Complete. Saved {len(unique_img_paths)} .pt files to {img_emb_dir}")

    # 4. Simulate Task 3: Generate Text Embeddings (Per Row)
    print(f"Simulating Task 3: Processing {len(df)} text instructions...")
    
    for idx, row in df.iterrows():
        # Save as RowIndex.pt
        save_path = os.path.join(text_emb_dir, f"{idx}.pt")
        
        # Mock BERT vector (768,)
        tensor = torch.randn(768, dtype=torch.float32)
        torch.save(tensor, save_path)
        
    print(f"Task 3 Complete. Saved {len(df)} .pt files to {text_emb_dir}")
    print("Done. Virtual offline data is ready.")

if __name__ == "__main__":
    generate_offline_features()
