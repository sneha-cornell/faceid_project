import tensorflow as tf
import time
import os
from ultra_light_model_tf import create_model, count_parameters
from dataset_tf import create_train_val_datasets


def single_face_loss(y_true, y_pred):
    """Loss function for single face detection in TensorFlow"""
    # Focus on center prediction only (most important)
    # y_pred shape: [batch, 7, 7, 5]
    # y_true shape: [batch, 5]
    
    # Get center prediction
    center_pred = y_pred[:, 3, 3, :]  # Center of 7x7 grid
    
    # Apply sigmoid to predictions
    center_pred_sigmoid = tf.sigmoid(center_pred)
    
    # MSE loss on center prediction
    loss = tf.keras.losses.MeanSquaredError()(y_true, center_pred_sigmoid)
    
    return loss


def accuracy_metric(y_true, y_pred):
    """Custom accuracy metric for single face detection"""
    # Get center prediction
    center_pred = y_pred[:, 3, 3, :]
    center_pred_sigmoid = tf.sigmoid(center_pred)
    
    # Check if confidence is high enough
    confidence_mask = center_pred_sigmoid[:, 4] > 0.5
    
    # Check if center coordinates are close
    center_dist = tf.norm(center_pred_sigmoid[:, :2] - y_true[:, :2], axis=1)
    distance_mask = center_dist < 0.2
    
    # Combined accuracy
    correct = tf.logical_and(confidence_mask, distance_mask)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    return accuracy


def train_full_dataset_tf():
    """Train on full dataset with proper train/val split using TensorFlow"""
    
    print("🎯 TENSORFLOW TRAINING ON FULL DATASET")
    print("=" * 50)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU available: {len(gpus)} device(s)")
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  No GPU available, using CPU")
    
    # Create full dataset
    data_dir = '../data/lfw-deepfunneled/lfw-deepfunneled'
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        return None, 0.0
    
    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        data_dir, batch_size=32, img_size=112, val_split=0.2
    )
    
    # Create model
    model = create_model()
    params = count_parameters(model)
    
    print(f"✅ Model: {params:,} parameters ({params/1000:.1f}k)")
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss=single_face_loss,
        metrics=[accuracy_metric]
    )
    
    # Model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'full_dataset_model_tf.h5',
            monitor='val_accuracy_metric',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Training
    print(f"Training for 10 epochs...")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('full_dataset_model_tf_final.h5')
    
    # Get best accuracy
    best_accuracy = max(history.history['val_accuracy_metric']) * 100
    
    print(f"\n🎯 TensorFlow Training Complete!")
    print(f"📊 Best Accuracy: {best_accuracy:.1f}%")
    print(f"💾 Model saved: full_dataset_model_tf.h5")
    print(f"💾 Final model saved: full_dataset_model_tf_final.h5")
    
    return model, best_accuracy


if __name__ == '__main__':
    model, accuracy = train_full_dataset_tf() 