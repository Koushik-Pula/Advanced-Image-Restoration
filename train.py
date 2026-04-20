import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

from models.airnet_lite import AirNetLite

# --- 1. PYTORCH DATALOADER ---
class MixedDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, patch_size=256):
        self.deg_dir, self.clean_dir = degraded_dir, clean_dir
        self.files = [f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg'))]
        self.size = patch_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        deg = Image.open(os.path.join(self.deg_dir, self.files[idx])).convert('RGB')
        clean = Image.open(os.path.join(self.clean_dir, self.files[idx])).convert('RGB')
        
        i, j, h, w = transforms.RandomCrop.get_params(deg, output_size=(self.size, self.size))
        deg, clean = transforms.functional.crop(deg, i, j, h, w), transforms.functional.crop(clean, i, j, h, w)
        if torch.rand(1).item() > 0.5:
            deg, clean = transforms.functional.hflip(deg), transforms.functional.hflip(clean)
            
        return self.to_tensor(deg), self.to_tensor(clean)

# --- 2. EARLY STOPPING LOGIC ---
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"\n[EarlyStopping] Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
            self.counter = 0

# --- 3. VGG PERCEPTUAL LOSS ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # Extract features from multiple depths of a pre-trained VGG16
        blocks = []
        vgg_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        blocks.append(vgg_features[:4].eval())
        blocks.append(vgg_features[4:9].eval())
        blocks.append(vgg_features[9:16].eval())
        blocks.append(vgg_features[16:23].eval())
        
        # Freeze VGG parameters so we don't accidentally train it
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = nn.ModuleList(blocks)
        
        # ImageNet normalization statistics required by VGG
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            input = nn.functional.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = nn.functional.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x, y = input, target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += nn.functional.mse_loss(x, y)
            
        return loss

# --- 4. TRAINING LOOP ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Booting Perceptual AirNet-Lite on {device}...")

    # Artificially 4x the dataset size via patches
    train_dataset = torch.utils.data.ConcatDataset([
        MixedDataset("data/train_mixed_degraded", "data/DIV2K_train_HR", patch_size=256) 
        for _ in range(4) 
    ])
    val_dataset = MixedDataset("data/val_degraded", "data/val_clean", patch_size=256)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = AirNetLite().to(device)
    
    # Instantiate Compound Loss
    criterion_l1 = nn.L1Loss().to(device)
    criterion_vgg = VGGPerceptualLoss().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    early_stopper = EarlyStopping(patience=7)
    
    epochs = 100
    os.makedirs("weights", exist_ok=True)

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for deg, clean in pbar:
            deg, clean = deg.to(device), clean.to(device)
            optimizer.zero_grad()
            
            restored = model(deg)
            
            # Calculate Structural (L1) and Texture (VGG) Losses
            loss_l1 = criterion_l1(restored, clean)
            loss_vgg = criterion_vgg(restored, clean)
            
            # Combine: 1.0 weight for structure, 0.1 weight for perceptual texture hallucination
            loss = loss_l1 + (0.1 * loss_vgg)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(total_loss=loss.item(), l1=loss_l1.item(), vgg=loss_vgg.item())
            
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for deg, clean in val_loader:
                deg, clean = deg.to(device), clean.to(device)
                restored = model(deg)
                
                v_l1 = criterion_l1(restored, clean)
                v_vgg = criterion_vgg(restored, clean)
                val_loss += (v_l1 + (0.1 * v_vgg)).item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- EARLY STOPPING CHECK ---
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("\n🚨 Overfitting Detected! Halting Training.")
            print(f"Restoring best weights from epoch {epoch + 1 - early_stopper.patience}.")
            torch.save(early_stopper.best_weights, "weights/airnet_lite_best.pth")
            break

        # Save checkpoint anyway just in case
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"weights/airnet_lite_ep{epoch+1}.pth")

    print("\nTraining Complete. Final SOTA weights saved to weights/airnet_lite_best.pth")

if __name__ == "__main__":
    train()