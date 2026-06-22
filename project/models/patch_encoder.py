import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEncoder3D(nn.Module):
    def __init__(self):
        super(PatchEncoder3D, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).view(x.size(0), -1)
        
        local_feat = x
        out = self.fc(local_feat)
        out = F.normalize(out, p=2, dim=1)
        
        return out
