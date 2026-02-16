import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from models.cnn_model import CNNModel
class ImageDataset(Dataset):
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
def create_best_model():
    """
    Create and train the best CNN model for copy-move forgery detection.
    """
    print("Creating the best model for copy-move forgery detection...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_paths = []
    compression_labels = []
    forgery_labels = []
    copy_move_labels = []
    sample_images = {
        'static/samples/original.jpg': {'compression': 0, 'forgery': 0, 'copy_move': 0},
        'static/samples/jpeg_compressed.jpg': {'compression': 1, 'forgery': 0, 'copy_move': 0},
        'static/samples/copy_move_forgery.jpg': {'compression': 0, 'forgery': 1, 'copy_move': 1},
        'static/samples/splicing_forgery.jpg': {'compression': 1, 'forgery': 1, 'copy_move': 0},
    }
    upload_dir = 'static/uploads'
    if os.path.exists(upload_dir):
        for filename in os.listdir(upload_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                filepath = os.path.join(upload_dir, filename)
                if 'original' in filename.lower():
                    compression_labels.append(0)
                    forgery_labels.append(0)
                    copy_move_labels.append(0)
                elif 'jpeg' in filename.lower() or 'compressed' in filename.lower():
                    compression_labels.append(1)
                    forgery_labels.append(0)
                    copy_move_labels.append(0)
                elif 'forgery' in filename.lower() or 'fake' in filename.lower():
                    compression_labels.append(0)
                    forgery_labels.append(1)
                    copy_move_labels.append(0)
                elif 'copy_move' in filename.lower():
                    compression_labels.append(0)
                    forgery_labels.append(1)
                    copy_move_labels.append(1)
                else:
                    compression_labels.append(2)
                    forgery_labels.append(0)
                    copy_move_labels.append(0)
                image_paths.append(filepath)
    for img_path, labels_dict in sample_images.items():
        if os.path.exists(img_path):
            image_paths.append(img_path)
            compression_labels.append(labels_dict['compression'])
            forgery_labels.append(labels_dict['forgery'])
            copy_move_labels.append(labels_dict['copy_move'])
    print(f"Found {len(image_paths)} images for training:")
    for img in image_paths:
        print(f"  - {img}")
    if not image_paths:
        print("No images found for training. Creating a pre-trained model with random weights.")
        model = CNNModel()
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/cnn_model.pth')
        print("Best model created and saved with random weights.")
        return
    labels = list(zip(compression_labels, forgery_labels, copy_move_labels))
    dataset = ImageDataset(image_paths, labels, transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    print(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            compression_out, forgery_out, copy_move_out = model(inputs)
            compression_labels, forgery_labels, copy_move_labels = zip(*labels)
            compression_labels = torch.tensor(compression_labels)
            forgery_labels = torch.tensor(forgery_labels)
            copy_move_labels = torch.tensor(copy_move_labels)
            loss_compression = criterion(compression_out, compression_labels)
            loss_forgery = criterion(forgery_out, forgery_labels)
            loss_copy_move = criterion(copy_move_out, copy_move_labels)
            loss = loss_compression + loss_forgery + loss_copy_move
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/cnn_model.pth')
    print("Best CNN model trained and saved successfully!")
if __name__ == "__main__":
    create_best_model()