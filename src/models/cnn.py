"""Simple CNN models for federated learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""
    
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        hidden_channels: int = 32,
    ):
        """
        Initialize Simple CNN.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=5, padding=2)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(hidden_channels * 4 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features before the final classification layer."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        return x


class TwoLayerCNN(nn.Module):
    """Two-layer CNN for FEMNIST classification."""
    
    def __init__(
        self,
        num_classes: int = 62,
        in_channels: int = 1,
        hidden_channels: int = 32,
    ):
        """
        Initialize Two-layer CNN.
        
        Args:
            num_classes: Number of output classes (62 for FEMNIST)
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(hidden_channels * 2 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features before the final classification layer."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        return x


class CNNMnist(nn.Module):
    """CNN optimized for MNIST-like datasets."""
    
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
    ):
        """
        Initialize CNN for MNIST.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
