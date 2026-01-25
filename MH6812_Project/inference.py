import torch
import argparse
from fusion_model import ImageTextFusionModel
from vision_encoder import get_image_embedding
from text_encoder import get_text_embedding

def predict(image_path, instruction, model_path='model_final.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    model = ImageTextFusionModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return

    # 2. 获取特征
    print(f"Processing image: {image_path}")
    img_emb = get_image_embedding(image_path).to(device)
    
    print(f"Processing text: {instruction}")
    text_emb = get_text_embedding(instruction).to(device)
    
    # 3. 推理
    with torch.no_grad():
        # Add batch dimension
        img_emb = img_emb.unsqueeze(0)
        text_emb = text_emb.unsqueeze(0)
        
        output = model(img_emb, text_emb)
        
    params = output.squeeze().tolist()
    print("\n--- Prediction Result ---")
    print(f"Brightness Adjustment: {params[0]:.4f}")
    print(f"Contrast Adjustment:   {params[1]:.4f}")
    print(f"Temperature Adjustment:{params[2]:.4f}")
    
    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MH6812 Project Inference')
    parser.add_argument('--image', type=str, default='test_image.jpg', help='Path to the input image')
    parser.add_argument('--text', type=str, default='Make it warmer', help='Editing instruction')
    args = parser.parse_args()
    
    predict(args.image, args.text)
