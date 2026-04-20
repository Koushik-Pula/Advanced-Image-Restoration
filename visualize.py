import os
import torch
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from models.airnet_lite import AirNetLite

def generate_comparison_grid(degraded_dir, clean_dir, weights_path, output_path, num_samples=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Presentation Grid on {device}...")

    # Load Model
    model = AirNetLite().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Get random sample of files from the validation set
    all_files = [f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg'))]
    sample_files = random.sample(all_files, min(num_samples, len(all_files)))
    
    to_tensor = transforms.ToTensor()
    
    # Setup Matplotlib Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    # Column Titles
    titles = ["Degraded Input (Unseen)", "AirNet-Lite Restoration", "Clean Ground Truth"]
    for j in range(3):
        axes[0, j].set_title(titles[j], fontsize=14, fontweight='bold', pad=10)

    with torch.no_grad():
        for i, file_name in enumerate(sample_files):
            # Load images
            deg_img = Image.open(os.path.join(degraded_dir, file_name)).convert('RGB')
            clean_img = Image.open(os.path.join(clean_dir, file_name)).convert('RGB')

            # Crop to 256x256 so the grid looks uniform
            deg_patch = transforms.functional.center_crop(deg_img, (256, 256))
            clean_patch = transforms.functional.center_crop(clean_img, (256, 256))

            # Run Inference
            img_tensor = to_tensor(deg_patch).unsqueeze(0).to(device)
            restored_tensor = model(img_tensor).squeeze().cpu()
            
            restored_img = transforms.ToPILImage()(restored_tensor)

            # Plot row
            images = [deg_patch, restored_img, clean_patch]
            for j in range(3):
                axes[i, j].imshow(images[j])
                axes[i, j].axis('off')

    # Save the high-res graphic
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Success! High-resolution presentation grid saved to {output_path}")

if __name__ == "__main__":
    generate_comparison_grid(
        degraded_dir="data/val_degraded",
        clean_dir="data/val_clean",
        weights_path="weights/airnet_lite_best.pth",
        output_path="data/presentation_grid.png",
        num_samples=4
    )