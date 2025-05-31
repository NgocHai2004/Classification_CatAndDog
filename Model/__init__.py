import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_Cat_Dog(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  
        )
        
        self.flatten = nn.Flatten()

        # Cập nhật tham số đầu vào cho lớp fully connected
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 28 * 28, 256),  # Thay 14336 thành 64 * 28 * 28 = 50176
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layer(x)
        return x
