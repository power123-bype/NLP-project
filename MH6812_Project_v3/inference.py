import torch
import argparse
import os
import numpy as np
from PIL import Image, ImageEnhance
from fusion_model import ImageTextFusionModel

# Import directly from the task files as per project structure
from task2 import get_image_embedding
from task3 import get_text_embedding

def apply_edits(image_path, brightness_factor, contrast_factor, temp_factor, output_path=None):
    """
    Apply predicted adjustments to the image.
    
    Args:
        image_path: Path to the input image.
        brightness_factor: Predicted brightness adjustment (normalized).
        contrast_factor: Predicted contrast adjustment (normalized).
        temp_factor: Predicted temperature adjustment (normalized).
        output_path: Path to save the edited image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image for editing: {e}")
        return None

    # 1. Apply Brightness (Map normalized output to reasonable factor)
    # Assuming training target was: factor = 1.0 + label (if label is around 0)
    # Or strict factor directly if label was factor itself.
    # Based on search, labels were normalized. Let's assume prediction is additive to 1.0
    # E.g. pred 0.2 -> factor 1.2
    b_factor = 1.0 + brightness_factor
    img = ImageEnhance.Brightness(img).enhance(b_factor)
    
    # 2. Apply Contrast
    c_factor = 1.0 + contrast_factor
    img = ImageEnhance.Contrast(img).enhance(c_factor)
    
    # 3. Apply Temperature (Simulated)
    # Warm: Increase Red, Decrease Blue
    # Cold: Decrease Red, Increase Blue
    # Temp factor scale: Let's assume raw range was +/- 50, normalized to roughly +/- 1? 
    # Or just use the predicted value as a strength coefficient.
    
    # Convert to numpy for channel manipulation
    arr = np.array(img).astype(np.float32)
    
    # Strength of temperature effect. 
    # If prediction is 0.5 (warm), we add red and subtract blue.
    # Let's scale it to pixel values. e.g. 0.5 * 30 = 15 pixel shift
    t_scale = 30.0 
    shift = temp_factor * t_scale
    
    if shift > 0: # Warmer
        arr[:, :, 0] += shift # R
        arr[:, :, 2] -= shift # B
    else: # Cooler
        arr[:, :, 0] += shift # R (shift is negative, so subtracts)
        arr[:, :, 2] -= shift # B (shift is negative, so adds)
        
    # Clip values
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    
    if output_path:
        img.save(output_path)
        print(f"Saved edited image to: {output_path}")
        
    return img

def predict(image_path, instruction, model_path='best_model.pth', output_image='output_edited.jpg'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载模型
    if not os.path.exists(model_path):
        if os.path.exists('model_final.pth'):
            print(f"'{model_path}' not found, falling back to 'model_final.pth'...")
            model_path = 'model_final.pth'
        else:
            print(f"Error: Model file '{model_path}' not found. Please train the model first.")
            return

    model = ImageTextFusionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # 2. 获取特征
    print(f"Processing image: {image_path}")
    try:
        img_emb = get_image_embedding(image_path).to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return
    
    print(f"Processing text: {instruction}")
    try:
        text_emb = get_text_embedding(instruction).to(device)
    except Exception as e:
        print(f"Error processing text: {e}")
        return
    
    # 3. 推理
    with torch.no_grad():
        # Add batch dimension
        img_emb = img_emb.unsqueeze(0)
        text_emb = text_emb.unsqueeze(0)
        
        output = model(img_emb, text_emb)
        
    params = output.squeeze().tolist()
    
    b_val, c_val, t_val = params[0], params[1], params[2]
    
    print("\n--- Prediction Result ---")
    print(f"Brightness Adjustment: {b_val:.4f}")
    print(f"Contrast Adjustment:   {c_val:.4f}")
    print(f"Temperature Adjustment:{t_val:.4f}")
    
    # 4. 生成可视化结果
    apply_edits(image_path, b_val, c_val, t_val, output_path=output_image)
    
    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MH6812 Project Inference')
    parser.add_argument('--image', type=str, default='dataset/images/original/000000133233.jpg', help='Path to the input image')
    parser.add_argument('--text', type=str, default='Make it brighter and warmer', help='Editing instruction')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to the model file')
    parser.add_argument('--output', type=str, default='output_edited.jpg', help='Path to save the output image')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Warning: Default image {args.image} not found. Please specify --image path.")
    
    predict(args.image, args.text, args.model, args.output)
