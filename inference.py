import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.airnet_lite import AirNetLite

def run_inference(input_path, output_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running All-in-One Restoration on {device}...")

    # Load Model
    model = AirNetLite().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load and format image for U-Net (dimensions must be divisible by 8)
    raw_img = Image.open(input_path).convert("RGB")
    w, h = raw_img.size
    new_w, new_h = w - (w % 8), h - (h % 8)
    raw_img = raw_img.resize((new_w, new_h))
    
    img_tensor = transforms.ToTensor()(raw_img).unsqueeze(0).to(device)

    # Hallucinate the restoration
    with torch.no_grad():
        restored_tensor = model(img_tensor)

    # Convert back to PIL Image
    restored_img = transforms.ToPILImage()(restored_tensor.squeeze().cpu())

    # Create Side-by-Side Comparison for the Presentation
    comparison = Image.new('RGB', (new_w * 2, new_h))
    comparison.paste(raw_img, (0, 0))
    comparison.paste(restored_img, (new_w, 0))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison.save(output_path)
    print(f"Success! Side-by-side comparison saved to {output_path}")

if __name__ == "__main__":
    # Pointing to an unseen image from the validation set
    # IN_IMG = os.path.join("data", "val_degraded", os.listdir("data/val_degraded")[0])
    IN_IMG = os.path.join("data","degraded.png")
    OUT_IMG = "data/restored_comparison.png"
    WEIGHTS = "weights/airnet_lite_best.pth"
    
    run_inference(IN_IMG, OUT_IMG, WEIGHTS)