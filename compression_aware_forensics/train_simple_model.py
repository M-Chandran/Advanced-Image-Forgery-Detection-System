#!/usr/bin/env python3
"""
Simple training script for the CNN model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from models.cnn_model import SimpleCNN
from utils.image_processing import preprocess_image
from PIL import Image
def create_dummy_dataset():
    """Create a simple dummy dataset for training"""
    num_samples = 100
    images = []
    compression_labels = []
    forgery_labels = []
    copy_move_labels = []
    for i in range(num_samples):
        img = np.random.rand(224, 224, 3).astype(np.float32)
        images.append(img)
        compression_labels.append(np.random.randint(0, 3))
        forgery_labels.append(np.random.randint(0, 2))
        copy_move_labels.append(np.random.randint(0, 2))
    images = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2)
    compression_labels = torch.from_numpy(np.array(compression_labels))
    forgery_labels = torch.from_numpy(np.array(forgery_labels))
    copy_move_labels = torch.from_numpy(np.array(copy_move_labels))
    return images, compression_labels, forgery_labels, copy_move_labels
def train_model():
    """Train the simple CNN model"""
    print("Creating model...")
    model = SimpleCNN()
    print("Creating dummy dataset...")
    images, comp_labels, forg_labels, cm_labels = create_dummy_dataset()
    dataset = TensorDataset(images, comp_labels, forg_labels, cm_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Training model...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_images, batch_comp, batch_forg, batch_cm in dataloader:
            optimizer.zero_grad()
            comp_out, forg_out, cm_out = model(batch_images)
            loss_comp = criterion(comp_out, batch_comp)
            loss_forg = criterion(forg_out, batch_forg)
            loss_cm = criterion(cm_out, batch_cm)
            loss = loss_comp + loss_forg + loss_cm
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(dataloader):.4f}")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/cnn_model.pth')
    print("Model saved to models/cnn_model.pth")
    print("Testing model...")
    model.eval()
    with torch.no_grad():
        test_images = images[:5]
        comp_out, forg_out, cm_out = model(test_images)
        _, comp_pred = torch.max(comp_out, 1)
        _, forg_pred = torch.max(forg_out, 1)
        _, cm_pred = torch.max(cm_out, 1)
        print("Test predictions:")
        for i in range(5):
            print(f"  Sample {i+1}: Comp={comp_pred[i].item()}, Forg={forg_pred[i].item()}, CM={cm_pred[i].item()}")
if __name__ == '__main__':
    train_model()