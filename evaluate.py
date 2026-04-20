import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

from models.airnet_lite import AirNetLite

def evaluate_model(degraded_dir, clean_dir, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculating Out-of-Sample PSNR and SSIM on {device}...")

    # Load Model
    model = AirNetLite().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    files = [f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg'))]
    to_tensor = transforms.ToTensor()
    
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for f in tqdm(files, desc="Evaluating Validation Set"):
            deg_img = Image.open(os.path.join(degraded_dir, f)).convert('RGB')
            clean_img = Image.open(os.path.join(clean_dir, f)).convert('RGB')

            # Crop to standard size for uniform evaluation
            deg_patch = transforms.functional.center_crop(deg_img, (256, 256))
            clean_patch = transforms.functional.center_crop(clean_img, (256, 256))

            img_tensor = to_tensor(deg_patch).unsqueeze(0).to(device)
            restored_tensor = model(img_tensor).squeeze().cpu().numpy()
            
            # Convert tensors to numpy arrays (H, W, C) for Scikit-Image
            restored_np = np.transpose(restored_tensor, (1, 2, 0))
            clean_np = np.transpose(to_tensor(clean_patch).numpy(), (1, 2, 0))

            # Calculate Metrics
            total_psnr += calculate_psnr(clean_np, restored_np, data_range=1.0)
            total_ssim += calculate_ssim(clean_np, restored_np, data_range=1.0, channel_axis=-1)

    avg_psnr = total_psnr / len(files)
    avg_ssim = total_ssim / len(files)

    print("\n" + "="*40)
    print("🏆 FINAL VALIDATION METRICS 🏆")
    print("="*40)
    print(f"Average PSNR: {avg_psnr:.2f} dB (Higher is better)")
    print(f"Average SSIM: {avg_ssim:.4f} (Closer to 1.0 is better)")
    print("="*40)

if __name__ == "__main__":
    evaluate_model(
        degraded_dir="data/val_degraded",
        clean_dir="data/val_clean",
        weights_path="weights/airnet_lite_best.pth"
    )