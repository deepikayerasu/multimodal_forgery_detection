# models/video_model.py
import torch
import torch.nn as nn

class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Tiny CNN that works offline (no pretrained weights needed)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 299 -> 149
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 149 -> 74
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, 256)

    def forward(self, x):
        x = self.features(x)          # [B,64,1,1]
        x = x.view(x.size(0), -1)     # [B,64]
        x = self.fc(x)                # [B,256]
        return x
