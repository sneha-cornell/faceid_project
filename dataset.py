import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset


class CustomLFWDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        person_dirs.sort()
        
        # Create class mapping
        for idx, person in enumerate(person_dirs):
            self.class_to_idx[person] = idx
            self.classes.append(person)
        
        # Collect all image paths and labels
        for person in person_dirs:
            person_dir = os.path.join(data_dir, person)
            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[person]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_lfw_dataloaders(data_dir='data/lfw-deepfunneled/lfw-deepfunneled', batch_size=32, img_size=96, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = CustomLFWDataset(data_dir=data_dir, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Dataset loaded: {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"Train: {train_size} images, Val: {val_size} images")
    
    return train_loader, val_loader, dataset.classes 