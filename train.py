import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
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

# --- 3. TRAINING LOOP ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Booting Self-Monitoring AirNet-Lite on {device}...")

    # Loaders: Notice we now extract 4 patches per image to artificially 4x the dataset size!
    train_dataset = torch.utils.data.ConcatDataset([
        MixedDataset("data/train_mixed_degraded", "data/DIV2K_train_HR", patch_size=256) 
        for _ in range(4) 
    ])
    val_dataset = MixedDataset("data/val_degraded", "data/val_clean", patch_size=256)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = AirNetLite().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    
    early_stopper = EarlyStopping(patience=7)
    epochs = 100 # Set high safely; Early Stopping will catch it
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
            loss = criterion(restored, clean)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for deg, clean in val_loader:
                deg, clean = deg.to(device), clean.to(device)
                restored = model(deg)
                val_loss += criterion(restored, clean).item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1} Summary -> Train L1: {avg_train_loss:.4f} | Val L1: {avg_val_loss:.4f}")

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