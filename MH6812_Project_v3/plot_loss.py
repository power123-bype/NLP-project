import matplotlib
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

print("Starting plot_loss.py...", flush=True)

def plot_loss(csv_file, output_file):
    print(f"Checking file: {csv_file}", flush=True)
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.", flush=True)
        return

    try:
        print("Reading CSV...", flush=True)
        df = pd.read_csv(csv_file)
        print(f"Read {len(df)} rows.", flush=True)
        
        print("Creating plot...", flush=True)
        plt.figure(figsize=(10, 6))
        
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', linestyle='-')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red', linestyle='--')
        
        plt.title('Training and Validation Loss Curve', fontsize=16)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        print(f"Saving to {output_file}...", flush=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Success: Loss curve saved to {output_file}", flush=True)
        
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_loss('/workspace/MH6812_Project/training_log.csv', '/workspace/MH6812_Project/loss_curve.png')
