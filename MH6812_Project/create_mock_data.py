import os
import csv
import random

def create_mock_data():
    """
    Create a mock metadata.csv and dataset folder structure for testing.
    """
    os.makedirs('dataset/images', exist_ok=True)
    
    # Create some dummy image files
    for i in range(5):
        with open(f'dataset/images/img_{i}.jpg', 'w') as f:
            f.write('dummy image content')

    # Create metadata.csv
    csv_file = 'metadata.csv'
    headers = ['original_img_path', 'processed_img_path', 'instruction', 'label_brightness', 'label_contrast', 'label_temp']
    
    data = []
    for i in range(10): # Generate 10 samples
        row = [
            f'dataset/images/img_{i}_orig.jpg',
            f'dataset/images/img_{i}_proc.jpg',
            f'Make the image look warmer and brighter version {i}',
            random.uniform(0, 1), # Normalized labels
            random.uniform(0, 1),
            random.uniform(0, 1)
        ]
        data.append(row)
        
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    
    print(f"Mock data created: {csv_file}")

if __name__ == "__main__":
    create_mock_data()
