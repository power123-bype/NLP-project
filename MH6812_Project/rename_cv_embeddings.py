import os
import pandas as pd
import shutil
from tqdm import tqdm

def main():
    # Paths
    BASE_DIR = "/workspace/MH6812_Project"
    CSV_PATH = os.path.join(BASE_DIR, "dataset/original_metadata.csv")
    SOURCE_DIR = os.path.join(BASE_DIR, "img")
    TARGET_DIR = os.path.join(BASE_DIR, "dataset/cv_embeddings")

    # Create target directory
    print(f"Creating target directory: {TARGET_DIR}")
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    # Read CSV
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file {CSV_PATH} does not exist.")
        return

    print(f"Reading {CSV_PATH}...")
    df = None
    encodings = ['utf-8', 'gb18030', 'latin1', 'cp1252']
    for enc in encodings:
        try:
            print(f"Trying encoding: {enc}")
            df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip", encoding=enc)
            print(f"Successfully read with encoding: {enc}")
            break
        except Exception as e:
            print(f"Failed with encoding {enc}: {e}")
    
    if df is None:
        print("Error: Could not read CSV with any of the attempted encodings.")
        return

    print(f"Processing {len(df)} rows...")
    success_count = 0
    missing_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Source file: img/{idx}.pt
        source_file = os.path.join(SOURCE_DIR, f"{idx}.pt")
        
        if not os.path.exists(source_file):
            missing_count += 1
            continue
            
        # Target file: ID from original_img_path
        # original_img_path example: dataset/images/original/000000133233.jpg
        try:
            original_path = row['original_img_path']
            filename = os.path.basename(original_path) # 000000133233.jpg
            image_id = os.path.splitext(filename)[0]   # 000000133233
            
            target_file = os.path.join(TARGET_DIR, f"{image_id}.pt")
            
            # Copy file
            shutil.copy2(source_file, target_file)
            success_count += 1
        except Exception as e:
            print(f"Error processing row {idx}: {e}")

    print(f"Done. Successfully renamed and copied: {success_count} files.")
    print(f"Missing source files: {missing_count}")

if __name__ == "__main__":
    main()
