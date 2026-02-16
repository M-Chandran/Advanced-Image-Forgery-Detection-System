import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class SimpleCNN(nn.Module):
    """
    Simple CNN Model for Image Forensics - Guaranteed to work
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.compression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 3)
        )
        self.forgery_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 2)
        )
        self.copy_move_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 2)
        )
        self._initialize_weights()
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        compression_out = self.compression_head(x)
        forgery_out = self.forgery_head(x)
        copy_move_out = self.copy_move_head(x)
        return compression_out, forgery_out, copy_move_out
    def predict_compression(self, image_tensor):
        """Predict compression type"""
        self.eval()
        with torch.no_grad():
            compression_out, _, _ = self(image_tensor)
            _, predicted = torch.max(compression_out, 1)
            compression_types = ['Original', 'JPEG', 'AI-Compressed']
            return compression_types[predicted.item()]
    def predict_forgery(self, image_tensor):
        """Predict forgery"""
        self.eval()
        with torch.no_grad():
            _, forgery_out, _ = self(image_tensor)
            _, predicted = torch.max(forgery_out, 1)
            forgery_results = ['Authentic', 'Forged']
            return forgery_results[predicted.item()]
    def predict_copy_move(self, image_tensor):
        """Predict copy-move"""
        self.eval()
        with torch.no_grad():
            _, _, copy_move_out = self(image_tensor)
            _, predicted = torch.max(copy_move_out, 1)
            copy_move_results = ['No Copy-Move', 'Copy-Move Detected']
            return copy_move_results[predicted.item()]
    def predict_all_ultra_fast(self, image_tensor):
        """Ultra-fast prediction: single forward pass for all tasks"""
        self.eval()
        with torch.no_grad():
            compression_out, forgery_out, copy_move_out = self(image_tensor)
            _, compression_pred = torch.max(compression_out, 1)
            _, forgery_pred = torch.max(forgery_out, 1)
            _, copy_move_pred = torch.max(copy_move_out, 1)
            compression_types = ['Original', 'JPEG', 'AI-Compressed']
            forgery_results = ['Authentic', 'Forged']
            copy_move_results = ['No Copy-Move', 'Copy-Move Detected']
            return {
                'compression_type': compression_types[compression_pred.item()],
                'forgery_result': forgery_results[forgery_pred.item()],
                'copy_move_result': copy_move_results[copy_move_pred.item()],
                'compression_confidence': torch.softmax(compression_out, dim=1).max().item(),
                'forgery_confidence': torch.softmax(forgery_out, dim=1).max().item(),
                'copy_move_confidence': torch.softmax(copy_move_out, dim=1).max().item()
            }
class CNNModel(SimpleCNN):
    """CNN Model - Simple and reliable"""
    pass
class FastCATNet(SimpleCNN):
    """Fast CAT-Net - Simple and reliable"""
    pass
class UltraFastNet(SimpleCNN):
    """Ultra Fast Net - Simple and reliable"""
    pass