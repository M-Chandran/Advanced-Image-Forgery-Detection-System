import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class UltraLightModel(nn.Module):
    """
    Ultra-Lightweight Model: Maximum speed optimization for real-time inference
    Extremely minimal architecture designed for 1-2 second inference
    """
    def __init__(self):
        super(UltraLightModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.compression_head = nn.Linear(32, 3)
        self.forgery_head = nn.Linear(32, 2)
        self.copy_move_head = nn.Linear(32, 2)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        compression_out = self.compression_head(x)
        forgery_out = self.forgery_head(x)
        copy_move_out = self.copy_move_head(x)
        return compression_out, forgery_out, copy_move_out
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
class LightningFastModel(nn.Module):
    """
    Lightning Fast Model: Minimal possible architecture for instant inference
    """
    def __init__(self):
        super(LightningFastModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.compression_head = nn.Linear(8, 3)
        self.forgery_head = nn.Linear(8, 2)
        self.copy_move_head = nn.Linear(8, 2)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        compression_out = self.compression_head(x)
        forgery_out = self.forgery_head(x)
        copy_move_out = self.copy_move_head(x)
        return compression_out, forgery_out, copy_move_out
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