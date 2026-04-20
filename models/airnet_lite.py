import torch
import torch.nn as nn
import torchvision.models as models

class ScoutEncoder(nn.Module):
    def __init__(self, vector_size=64):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, vector_size)

    def forward(self, x):
        f = self.features(x).view(x.size(0), -1)
        return self.fc(f)

class AirNetLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.scout = ScoutEncoder(vector_size=64)
        
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        
        # Switchboard Bottleneck
        self.vector_projector = nn.Linear(64, 256)
        self.bottleneck = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        
        # Decoder
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU())
        
        self.final = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        deg_vector = self.scout(x)
        
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Modulate features using the scout's findings
        deg_weights = self.vector_projector(deg_vector).unsqueeze(-1).unsqueeze(-1)
        b = self.bottleneck(e3) * deg_weights 
        
        d1 = self.dec1(b)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        
        return torch.sigmoid(self.final(d3))