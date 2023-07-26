import torch
import torchvision
from torch import nn


"""ResNet18(2,3)"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                nn.BatchNorm2d(out_channels)
                            )
        else:
            self.identity = nn.Identity()
        
    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        
        # 根据resnet18结构定义网络层
        self.net = nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    ResidualBlock(64, 64, stride=1),
                    ResidualBlock(64, 64, stride=1),
                    ResidualBlock(64, 128, stride=2),
                    ResidualBlock(128, 128, stride=1),
                    ResidualBlock(128, 256, stride=2),
                    ResidualBlock(256, 256, stride=1),
                    ResidualBlock(256, 512, stride=2),
                    ResidualBlock(512, 512, stride=1),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, num_classes)
                )
        
    def forward(self, x):
        return self.net(x)