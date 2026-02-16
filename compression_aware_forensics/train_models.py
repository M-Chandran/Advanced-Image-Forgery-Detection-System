import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.cnn_model import CNNModel
from models.autoencoder import Autoencoder
import urllib.request
import zipfile
import shutil
class CASIAv2Dataset(Dataset):
    """
    CASIA v2 Dataset for image forgery detection
    """
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels = []
        if train:
            authentic_dir = os.path.join(root_dir, 'Au')
            forged_dir = os.path.join(root_dir, 'Tp')
        else:
            authentic_dir = os.path.join(root_dir, 'Au_test')
            forged_dir = os.path.join(root_dir, 'Tp_test')
        if os.path.exists(authentic_dir):
            for img_name in os.listdir(authentic_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    self.image_paths.append(os.path.join(authentic_dir, img_name))
                    self.labels.append(0)
        if os.path.exists(forged_dir):
            for img_name in os.listdir(forged_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    self.image_paths.append(os.path.join(forged_dir, img_name))
                    self.labels.append(1)
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label
def download_casia_v2(dataset_path):
    """
    Download and extract CASIA v2 dataset
    """
    print("CASIA v2 dataset is a large dataset that needs to be downloaded manually.")
    print("Please download CASIA v2 from:")
    print("https://github.com/CASIA-IVA-Lab/CASIA2-groundtruth")
    print("or")
    print("https://www.kaggle.com/datasets/sophatvath/casia-20-image-tampering-detection-dataset")
    print("")
    print("Extract the dataset to:", dataset_path)
    print("Expected structure:")
    print(f"{dataset_path}/")
    print("  ├── Au/          # Authentic images")
    print("  ├── Tp/          # Tampered images")
    print("  ├── Au_test/     # Test authentic images")
    print("  └── Tp_test/     # Test tampered images")
    print("")
    return False
def create_balanced_dataset(dataset_path, max_samples_per_class=1000):
    """
    Create a balanced dataset from CASIA v2
    """
    print(f"Creating balanced dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        if not download_casia_v2(dataset_path):
            print("Dataset not found. Using sample images for training.")
            return None, None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CASIAv2Dataset(dataset_path, transform=transform, train=True)
    if len(train_dataset) == 0:
        print("No training images found. Using sample images.")
        return None, None
    authentic_indices = [i for i, label in enumerate(train_dataset.labels) if label == 0]
    forged_indices = [i for i, label in enumerate(train_dataset.labels) if label == 1]
    authentic_indices = authentic_indices[:max_samples_per_class]
    forged_indices = forged_indices[:max_samples_per_class]
    balanced_indices = authentic_indices + forged_indices
    np.random.shuffle(balanced_indices)
    balanced_paths = [train_dataset.image_paths[i] for i in balanced_indices]
    balanced_labels = [train_dataset.labels[i] for i in balanced_indices]
    balanced_dataset = ImageDataset(balanced_paths, balanced_labels, transform)
    train_size = int(0.8 * len(balanced_dataset))
    val_size = len(balanced_dataset) - train_size
    train_dataset, val_dataset = random_split(balanced_dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)} (Authentic: {sum(1 for i in balanced_labels[:train_size] if balanced_labels[i] == 0)}, Forged: {sum(1 for i in balanced_labels[:train_size] if balanced_labels[i] == 1)})")
    print(f"Validation samples: {len(val_dataset)}")
    return train_dataset, val_dataset
class ImageDataset(Dataset):
    """Fallback dataset for sample images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
def train_cnn_model(dataset_path="CASIAv2", num_epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train the CNN model using CASIA v2 dataset with advanced techniques
    """
    print("🚀 Starting CNN Model Training with CASIA v2 Dataset")
    print("=" * 60)
    train_dataset, val_dataset = create_balanced_dataset(dataset_path, max_samples_per_class=1500)
    if train_dataset is None or val_dataset is None:
        print("⚠️  CASIA v2 dataset not found. Falling back to sample images...")
        train_dataset, val_dataset = create_sample_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    print(f"📊 Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    print(f"🔧 Batch size: {batch_size}, Learning rate: {learning_rate}, Epochs: {num_epochs}")
    print("=" * 60)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            compression_out, forgery_out, copy_move_out = model(inputs)
            loss_compression = criterion(compression_out, labels)
            loss_forgery = criterion(forgery_out, labels)
            loss_copy_move = criterion(copy_move_out, labels)
            total_loss = 0.3 * loss_compression + 0.5 * loss_forgery + 0.2 * loss_copy_move
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            _, predicted = torch.max(forgery_out, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                compression_out, forgery_out, copy_move_out = model(inputs)
                loss_compression = criterion(compression_out, labels)
                loss_forgery = criterion(forgery_out, labels)
                loss_copy_move = criterion(copy_move_out, labels)
                total_loss = 0.3 * loss_compression + 0.5 * loss_forgery + 0.2 * loss_copy_move
                val_loss += total_loss.item()
                _, predicted = torch.max(forgery_out, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/cnn_model_best.pth')
        else:
            patience_counter += 1
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        with open('training_log.txt', 'a') as f:
            f.write(f"Epoch {epoch+1:2d}/{num_epochs} | "
                   f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}% | "
                   f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}% | "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    if os.path.exists('models/cnn_model_best.pth'):
        model.load_state_dict(torch.load('models/cnn_model_best.pth'))
        torch.save(model.state_dict(), 'models/cnn_model.pth')
        print("✅ Best model saved as 'models/cnn_model.pth'")
    print("\n🔍 Final Model Evaluation:")
    accuracy, precision, recall, f1 = evaluate_model(model, val_loader)
    with open('training_log.txt', 'a') as f:
        f.write(f"\nFinal Evaluation:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
    print("🎉 CNN Model Training Completed!")
    return model
def create_enhanced_sample_dataset(num_samples=1000):
    """
    Create enhanced synthetic dataset with data augmentation when CASIA v2 is not available
    """
    print(f"Creating enhanced synthetic dataset with {num_samples} samples...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    base_images = [
        ('static/samples/original.jpg', 0),
        ('static/samples/jpeg_compressed.jpg', 0),
        ('static/samples/copy_move_forgery.jpg', 1),
        ('static/samples/splicing_forgery.jpg', 1),
    ]
    image_paths = []
    labels = []
    for img_path, label in base_images:
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(label)
    if not image_paths:
        dummy_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = ImageDataset([], [], dummy_transform)
        return dataset, dataset
    augmented_paths = []
    augmented_labels = []
    for i in range(num_samples // len(image_paths)):
        for img_path, label in zip(image_paths, labels):
            augmented_paths.append(img_path)
            augmented_labels.append(label)
    augmented_paths = augmented_paths[:num_samples]
    augmented_labels = augmented_labels[:num_samples]
    dataset = AugmentedImageDataset(augmented_paths, augmented_labels, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
class AugmentedImageDataset(Dataset):
    """Dataset with on-the-fly augmentation"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
def train_autoencoder(dataset_path, num_epochs=10):
    """
    Train the Autoencoder for AI-based compression simulation.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_paths = []
    dataset = ImageDataset(image_paths, [0]*len(image_paths), transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, _ in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
    torch.save(model.state_dict(), 'models/autoencoder.pth')
    print("Autoencoder trained and saved.")
def evaluate_model(model, test_loader):
    """
    Evaluate the model and calculate metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1
def setup_casia_v2_dataset():
    """
    Setup CASIA v2 dataset for training
    """
    print("🔧 CASIA v2 Dataset Setup Guide")
    print("=" * 50)
    print("To use CASIA v2 dataset for training:")
    print("")
    print("1. Download the dataset from:")
    print("   https://github.com/CASIA-IVA-Lab/CASIA2-groundtruth")
    print("   or")
    print("   https://www.kaggle.com/datasets/sophatvath/casia-20-image-tampering-detection-dataset")
    print("")
    print("2. Extract the dataset to 'CASIAv2/' directory in the project root")
    print("")
    print("3. Expected structure:")
    print("   CASIAv2/")
    print("   ├── Au/          # Authentic training images")
    print("   ├── Tp/          # Tampered training images")
    print("   ├── Au_test/     # Authentic test images")
    print("   └── Tp_test/     # Tampered test images")
    print("")
    print("4. Run training: python train_models.py")
    print("=" * 50)
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_casia_v2_dataset()
        sys.exit(0)
    print("🚀 Advanced Image Forgery Detection Training")
    print("=" * 50)
    casia_path = "CASIAv2"
    if os.path.exists(casia_path):
        print("✅ CASIA v2 dataset found!")
        dataset_path = casia_path
    else:
        print("⚠️  CASIA v2 dataset not found. Using sample images for demonstration.")
        print("   Run 'python train_models.py --setup' for setup instructions.")
        dataset_path = None
    print("\n🧠 Training CNN Model...")
    trained_model = train_cnn_model(
        dataset_path=dataset_path,
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    print("\n🤖 Training Autoencoder...")
    train_autoencoder(dataset_path or "dataset/generated", num_epochs=20)
    print("\n🎉 Training completed! Models saved in 'models/' directory.")
    print("📊 Check 'training_log.txt' for detailed training logs.")