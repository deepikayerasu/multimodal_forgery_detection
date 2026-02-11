import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 0 = Real, 1 = Fake
        )

    def forward(self, v_feat, a_feat):
        fused = torch.cat((v_feat, a_feat), dim=1)
        return self.fc(fused)
