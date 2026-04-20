import torch
import torch.nn as nn
import torchvision.models as models

class ScoutEncoder(nn.Module):
    """
    The scout network that identifies the degradation type (Rain, Blur, Haze).
    Outputs a 64-dimensional latent vector representing the specific damage profile.
    """
    def __init__(self, vector_size=64):
        super().__init__()
        # Use a lightweight, pre-trained ResNet18 to analyze the texture
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Strip off the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Map the 512-dimensional output down to our 64-dimensional degradation vector
        self.fc = nn.Linear(512, vector_size)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        deg_vector = self.fc(features)
        return deg_vector # Shape: [Batch, 64]

class AirNetLite(nn.Module):
    """
    The Multi-Task U-Net architecture.
    Uses the Scout vector to modulate its bottleneck features, preventing Task Interference.
    """
    def __init__(self):
        super().__init__()
        
        # 1. The Degradation Scout
        self.scout = ScoutEncoder(vector_size=64)
        
        # 2. U-Net Encoder (Downsampling the image)
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        
        # 3. The Switchboard (Bottleneck Integration)
        # Projects the 64-length scout vector to match the 256-channel image features
        self.vector_projector = nn.Linear(64, 256)
        self.bottleneck = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        
        # 4. U-Net Decoder (Upsampling)
        # IMPORTANT FIX: Replaced ConvTranspose2d with Bilinear Interpolation + Conv2d
        # This prevents the overlapping "checkerboard" artifacts in the final image.
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(256, 128, 3, padding=1), 
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(128, 64, 3, padding=1), 
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(64, 32, 3, padding=1), 
            nn.ReLU()
        )
        
        # Final output layer mapping back to 3 RGB channels
        self.final = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        # --- Stage 1: Scout the damage ---
        deg_vector = self.scout(x)
        
        # --- Stage 2: Encode the image ---
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # --- Stage 3: The Switchboard Integration ---
        # Reshape the scout vector so it can multiply with the spatial image features
        deg_weights = self.vector_projector(deg_vector).unsqueeze(-1).unsqueeze(-1)
        
        # Modulate the bottleneck features based on what the scout found
        b = self.bottleneck(e3)
        b = b * deg_weights 
        
        # --- Stage 4: Decode the restored image ---
        d1 = self.dec1(b)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        
        # Sigmoid ensures output pixel values remain between 0.0 and 1.0
        out = self.final(d3)
        return torch.sigmoid(out)