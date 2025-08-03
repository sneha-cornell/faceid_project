import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_lfw_dataloaders(data_dir='data/lfw-deepfunneled', batch_size=32, img_size=96, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.LFWPeople(root=data_dir, split='train', download=False, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(dataset.classes)
    return train_loader, val_loader, dataset.classes 