import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ultra_light_model import MicroFaceDetector, count_parameters


class FullFaceDataset(Dataset):
    """Full dataset for single face detection - uses all available data"""
    
    def __init__(self, data_dir, img_size=112):
        self.img_size = img_size
        self.samples = []
        
        # Get all images from all people
        for person_dir in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person_dir)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    if img_file.endswith('.jpg'):
                        self.samples.append(os.path.join(person_path, img_file))
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"Full dataset: {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Target: 1 face in center [cx, cy, w, h, conf]
        target = torch.tensor([0.5, 0.5, 0.4, 0.5, 1.0], dtype=torch.float32)
        
        return image, target


def single_face_loss(predictions, targets):
    """Loss function for single face detection"""
    batch_size = targets.size(0)
    
    # Focus on center prediction only (most important)
    center_pred = predictions[:, :, 3, 3]  # Center of 7x7 grid
    center_pred_sigmoid = torch.sigmoid(center_pred)
    
    # Simple MSE loss on center prediction
    loss = nn.MSELoss()(center_pred_sigmoid, targets)
    
    return loss


def train_full_dataset():
    """Train on full dataset with proper train/val split"""
    
    print("🎯 TRAINING ON FULL DATASET")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create full dataset
    dataset = FullFaceDataset('data/lfw-deepfunneled/lfw-deepfunneled', img_size=112)
    
    # Train/val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Create model
    model = MicroFaceDetector().to(device)
    params = count_parameters(model)
    
    print(f"✅ Model: {params:,} parameters ({params/1000:.1f}k)")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    num_epochs = 10
    best_accuracy = 0.0
    patience_counter = 0
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = single_face_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_detections = 0
        total_detections = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                predictions = model(images)
                loss = single_face_loss(predictions, targets)
                val_loss += loss.item()
                
                # Calculate accuracy
                batch_size = images.size(0)
                for b in range(batch_size):
                    # Get center prediction
                    center_pred = torch.sigmoid(predictions[b, :, 3, 3])
                    target = targets[b]
                    
                    # Check if confidence is high enough
                    if center_pred[4] > 0.5:  # Confidence threshold
                        # Check if center coordinates are close
                        center_dist = torch.norm(center_pred[:2] - target[:2])
                        if center_dist < 0.2:  # Distance threshold
                            correct_detections += 1
                    total_detections += 1
        
        # Epoch summary
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = correct_detections / total_detections * 100
        epoch_time = time.time() - start_time
        
        print(f"✅ Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.1f}s")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Accuracy: {accuracy:.1f}% ({correct_detections}/{total_detections})")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'full_dataset_model.pth')
            print(f"  💾 New best model! Accuracy: {best_accuracy:.1f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 3:
            print(f"  🛑 Early stopping after {epoch+1} epochs")
            break
        
        scheduler.step(val_loss)
    
    print(f"\n🎯 Training Complete!")
    print(f"📊 Best Accuracy: {best_accuracy:.1f}%")
    print(f"💾 Model saved: full_dataset_model.pth")
    
    return model, best_accuracy


if __name__ == '__main__':
    model, accuracy = train_full_dataset() 