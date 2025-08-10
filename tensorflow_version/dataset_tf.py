import tensorflow as tf
import os
from PIL import Image
import numpy as np


class FullFaceDatasetTF:
    """Full dataset for single face detection in TensorFlow"""
    
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
        
        print(f"Full dataset: {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Resize
        image = tf.image.resize(image, (self.img_size, self.img_size))
        
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        return image
    
    def create_target(self):
        """Create target tensor for single face detection"""
        # Target: 1 face in center [cx, cy, w, h, conf]
        target = tf.constant([0.5, 0.5, 0.4, 0.5, 1.0], dtype=tf.float32)
        return target
    
    def __getitem__(self, idx):
        """Get a single sample"""
        img_path = self.samples[idx]
        image = self.load_and_preprocess_image(img_path)
        target = self.create_target()
        return image, target


def create_tf_dataset(data_dir, batch_size=32, img_size=112):
    """Create TensorFlow dataset with proper batching and shuffling"""
    
    # Create dataset object
    dataset_obj = FullFaceDatasetTF(data_dir, img_size)
    
    # Create TensorFlow dataset
    image_paths = tf.constant(dataset_obj.samples)
    
    def load_sample(image_path):
        """Load and preprocess a single sample"""
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Resize
        image = tf.image.resize(image, (img_size, img_size))
        
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Create target
        target = tf.constant([0.5, 0.5, 0.4, 0.5, 1.0], dtype=tf.float32)
        
        return image, target
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset


def create_train_val_datasets(data_dir, batch_size=32, img_size=112, val_split=0.2):
    """Create training and validation datasets"""
    
    # Create full dataset
    full_dataset = create_tf_dataset(data_dir, batch_size, img_size)
    
    # Get dataset size
    dataset_size = len(list(full_dataset))
    train_size = int((1 - val_split) * dataset_size)
    val_size = dataset_size - train_size
    
    print(f"Dataset size: {dataset_size}")
    print(f"Train size: {train_size}")
    print(f"Val size: {val_size}")
    
    # Split dataset
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    
    # Add batching and shuffling
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    
    return train_dataset, val_dataset


if __name__ == '__main__':
    # Test dataset creation
    data_dir = '../data/lfw-deepfunneled/lfw-deepfunneled'
    
    if os.path.exists(data_dir):
        print("Testing TensorFlow dataset creation...")
        
        # Create datasets
        train_dataset, val_dataset = create_train_val_datasets(data_dir, batch_size=8)
        
        print(f"✅ Train dataset created")
        print(f"✅ Val dataset created")
        
        # Test iteration
        for batch_idx, (images, targets) in enumerate(train_dataset.take(1)):
            print(f"📐 Batch {batch_idx + 1}:")
            print(f"   Images shape: {images.shape}")
            print(f"   Targets shape: {targets.shape}")
            print(f"   Image range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
            break
    else:
        print(f"⚠️  Dataset directory not found: {data_dir}")
        print("Please ensure the LFW dataset is available.") 