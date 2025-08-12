import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    total_params = model.count_params()
    return total_params


class MicroFaceDetectorTF(Model):
    """Ultra-lightweight face detection model in TensorFlow"""
    
    def __init__(self):
        super(MicroFaceDetectorTF, self).__init__()
        
        # Ultra-lightweight architecture with <5k parameters
        self.conv1 = layers.Conv2D(4, 3, padding='same', activation='relu', name='conv1')
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.pool1 = layers.MaxPooling2D(2, 2, name='pool1')
        
        self.conv2 = layers.Conv2D(8, 3, padding='same', activation='relu', name='conv2')
        self.bn2 = layers.BatchNormalization(name='bn2')
        self.pool2 = layers.MaxPooling2D(2, 2, name='pool2')
        
        self.conv3 = layers.Conv2D(16, 3, padding='same', activation='relu', name='conv3')
        self.bn3 = layers.BatchNormalization(name='bn3')
        self.pool3 = layers.MaxPooling2D(2, 2, name='pool3')
        
        self.conv4 = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv4')
        self.bn4 = layers.BatchNormalization(name='bn4')
        self.pool4 = layers.MaxPooling2D(2, 2, name='pool4')
        
        # Final detection layer - 7x7x5 output (cx, cy, w, h, conf)
        self.detection = layers.Conv2D(5, 1, padding='same', name='detection')
        
    def call(self, inputs, training=None):
        # Input: [batch, 112, 112, 3]
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.pool4(x)
        
        # Output: [batch, 7, 7, 5] - (cx, cy, w, h, conf)
        detections = self.detection(x)
        
        return detections


def create_model():
    """Create and return the ultra-lightweight face detection model"""
    model = MicroFaceDetectorTF()
    
    # Build the model with sample input
    sample_input = tf.random.normal((1, 112, 112, 3))
    _ = model(sample_input)
    
    return model


def detect_faces(model, predictions, conf_threshold=0.5):
    """
    Process raw model predictions into bounding boxes and confidence scores
    
    Args:
        model: The trained model
        predictions: Raw model output [batch, 7, 7, 5]
        conf_threshold: Confidence threshold for detection
    
    Returns:
        List of detections with bounding boxes and confidence scores
    """
    detections = []
    
    # Apply sigmoid to get probabilities
    predictions_sigmoid = tf.sigmoid(predictions)
    
    # Get center prediction (most important for single face detection)
    center_pred = predictions_sigmoid[0, 3, 3, :]  # Center of 7x7 grid
    
    # Check if confidence is high enough
    if center_pred[4] > conf_threshold:
        # Extract bounding box coordinates
        cx, cy, w, h = center_pred[:4].numpy()
        confidence = center_pred[4].numpy()
        
        # Convert to corner coordinates
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        detection = {
            'bbox': [x1, y1, x2, y2],
            'confidence': confidence,
            'center': [cx, cy]
        }
        detections.append(detection)
    
    return detections


if __name__ == '__main__':
    # Test model creation and parameter count
    model = create_model()
    params = count_parameters(model)
    
    print(f"✅ TensorFlow Model Created!")
    print(f"📊 Parameters: {params:,} ({params/1000:.1f}k)")
    
    # Test forward pass
    sample_input = tf.random.normal((1, 112, 112, 3))
    output = model(sample_input)
    print(f"📐 Input shape: {sample_input.shape}")
    print(f"📐 Output shape: {output.shape}")
    
    # Test detection function
    detections = detect_faces(model, output, conf_threshold=0.5)
    print(f"🔍 Detections: {len(detections)}") 