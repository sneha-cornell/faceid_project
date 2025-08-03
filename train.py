import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_lfw_dataloaders
from model import load_model


def train(num_epochs=10, batch_size=32, lr=1e-3, img_size=96, device='cuda' if torch.cuda.is_available() else 'cpu'):
    train_loader, val_loader, classes = get_lfw_dataloaders(batch_size=batch_size, img_size=img_size)
    num_classes = len(classes)
    model = load_model(num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model.")
    print(f"Training complete. Best Val Acc: {best_acc:.4f}")

if __name__ == '__main__':
    train() 