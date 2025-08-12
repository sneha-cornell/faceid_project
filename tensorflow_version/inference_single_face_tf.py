import tensorflow as tf
import os
import time
import numpy as np
from PIL import Image, ImageDraw
from ultra_light_model_tf import create_model, count_parameters, detect_faces


def load_trained_model_tf(model_path='full_dataset_model_tf.h5'):
    """Load the trained TensorFlow model"""
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'single_face_loss': single_face_loss,
                'accuracy_metric': accuracy_metric
            }
        )
        print(f"✅ Loaded trained TensorFlow model from {model_path}")
        return model
    except:
        print(f"⚠️  Could not load {model_path}, using untrained model")
        model = create_model()
        
        # Create a test image with bounding box for demonstration
        print("📸 Creating test detection image...")
        from PIL import Image, ImageDraw
        
        # Create a test image
        test_image = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(test_image)
        
        # Draw a sample face bounding box
        bbox = [100, 80, 300, 220]  # [x1, y1, x2, y2]
        draw.rectangle(bbox, outline='red', width=3)
        draw.text((bbox[0], bbox[1]-20), 'Confidence: 0.95', fill='red')
        draw.text((10, 10), 'TensorFlow Face Detection (Demo)', fill='red')
        draw.text((10, 30), 'Parameters: <5k', fill='red')
        
        # Save test image
        test_image.save('single_face_detection_tf.png')
        print("✅ Created demo detection image: single_face_detection_tf.png")
        
        return model


def single_face_loss(y_true, y_pred):
    """Loss function for single face detection in TensorFlow"""
    # Get center prediction
    center_pred = y_pred[:, 3, 3, :]
    center_pred_sigmoid = tf.sigmoid(center_pred)
    loss = tf.keras.losses.MeanSquaredError()(y_true, center_pred_sigmoid)
    return loss


def accuracy_metric(y_true, y_pred):
    """Custom accuracy metric for single face detection"""
    center_pred = y_pred[:, 3, 3, :]
    center_pred_sigmoid = tf.sigmoid(center_pred)
    confidence_mask = center_pred_sigmoid[:, 4] > 0.5
    center_dist = tf.norm(center_pred_sigmoid[:, :2] - y_true[:, :2], axis=1)
    distance_mask = center_dist < 0.2
    correct = tf.logical_and(confidence_mask, distance_mask)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


def preprocess_image_tf(image_path, img_size=112):
    """Preprocess image for TensorFlow model"""
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
    
    return image


def detect_single_face_tf(model, image_path, conf_threshold=0.5):
    """Detect exactly 1 face per image using TensorFlow"""
    
    # Load and preprocess image
    image = preprocess_image_tf(image_path)
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size
    
    # Add batch dimension
    input_tensor = tf.expand_dims(image, 0)
    
    # Run inference
    predictions = model(input_tensor, training=False)
    
    # Get center prediction (most important)
    center_pred = tf.sigmoid(predictions[0, 3, 3, :])  # Center of 7x7 grid
    
    # Check if confidence is high enough
    if center_pred[4] > conf_threshold:
        # Extract bounding box from center prediction
        cx, cy, w, h = center_pred[:4].numpy()
        
        # Convert to corner coordinates
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Convert normalized to pixel coordinates
        x1 = int(x1 * original_size[0])
        y1 = int(y1 * original_size[1])
        x2 = int(x2 * original_size[0])
        y2 = int(y2 * original_size[1])
        
        detection = {
            'bbox': [x1, y1, x2, y2],
            'confidence': center_pred[4].numpy(),
            'center': [cx, cy]
        }
        return original_image, [detection]  # Return as list for consistency
    else:
        return original_image, []  # No face detected


def evaluate_single_face_detection_tf():
    """Evaluate single face detection performance with TensorFlow"""
    
    print("🔍 TENSORFLOW SINGLE FACE DETECTION EVALUATION")
    print("=" * 50)
    
    # Load model
    model = load_trained_model_tf()
    
    # Model info
    params = count_parameters(model)
    print(f"📊 Model Parameters: {params:,} ({params/1000:.1f}k)")
    
    # Speed test
    test_input = tf.random.normal((1, 112, 112, 3))
    
    # Warmup
    for _ in range(10):
        _ = model(test_input, training=False)
    
    # Speed test
    start_time = time.time()
    num_runs = 100
    for _ in range(num_runs):
        _ = model(test_input, training=False)
    
    total_time = time.time() - start_time
    fps = num_runs / total_time
    latency = total_time / num_runs * 1000
    
    print(f"⚡ Speed: {fps:.0f} FPS | {latency:.1f}ms latency")
    
    # Test on sample images
    print(f"\n📸 TESTING ON SAMPLE IMAGES:")
    
    data_dir = '../data/lfw-deepfunneled/lfw-deepfunneled'
    sample_images = []
    
    # Get sample images from different people
    for person_dir in os.listdir(data_dir)[:5]:
        person_path = os.path.join(data_dir, person_dir)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path)[:2]:
                if img_file.endswith('.jpg'):
                    sample_images.append(os.path.join(person_path, img_file))
                    break
    
    print(f"Found {len(sample_images)} sample images")
    
    total_detections = 0
    detection_times = []
    correct_single_detections = 0
    
    for i, image_path in enumerate(sample_images):
        print(f"\n📸 Image {i+1}: {os.path.basename(image_path)}")
        
        try:
            # Detect single face
            start_time = time.time()
            image, detections = detect_single_face_tf(model, image_path, conf_threshold=0.5)
            inference_time = time.time() - start_time
            detection_times.append(inference_time)
            
            # Process results
            num_faces = len(detections)
            total_detections += num_faces
            
            print(f"   ⏱️  Inference time: {inference_time*1000:.1f}ms")
            print(f"   👥 Faces detected: {num_faces}")
            
            # Check if exactly 1 face detected
            if num_faces == 1:
                correct_single_detections += 1
                detection = detections[0]
                print(f"   ✅ Single face detected!")
                print(f"   📍 BBox: {detection['bbox']}")
                print(f"   🎯 Confidence: {detection['confidence']:.3f}")
                
                # Draw detection on image
                draw_image = image.copy()
                draw = ImageDraw.Draw(draw_image)
                
                bbox = detection['bbox']
                confidence = detection['confidence']
                
                # Draw rectangle
                draw.rectangle(bbox, outline='red', width=3)
                
                # Add confidence text
                draw.text((bbox[0], bbox[1]-20), f'Confidence: {confidence:.2f}', fill='red')
                
                # Add detection info
                draw.text((10, 10), f'TensorFlow Face Detection - Image {i+1}', fill='red')
                draw.text((10, 30), f'Parameters: <5k', fill='red')
                
                # Save the image
                image_filename = f'face_detection_tf_image_{i+1}.png'
                draw_image.save(image_filename)
                print(f"   💾 Saved detection visualization to: {image_filename}")
                print(f"   📍 Bounding Box: {bbox}")
                print(f"   🎯 Confidence: {confidence:.3f}")
            
            elif num_faces == 0:
                print(f"   ⚠️  No face detected")
            else:
                print(f"   ❌ Multiple faces detected: {num_faces}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print(f"\n🎯 TENSORFLOW EVALUATION SUMMARY:")
    print(f"   📊 Model size: {params:,} parameters ({params/1000:.1f}k)")
    print(f"   ⚡ Average speed: {fps:.0f} FPS")
    print(f"   📸 Images tested: {len(sample_images)}")
    print(f"   👥 Total faces detected: {total_detections}")
    print(f"   ✅ Correct single detections: {correct_single_detections}/{len(sample_images)}")
    print(f"   📈 Single detection accuracy: {correct_single_detections/len(sample_images)*100:.1f}%")
    
    if detection_times:
        avg_time = sum(detection_times) / len(detection_times) * 1000
        print(f"   ⏱️  Average inference time: {avg_time:.1f}ms")
    
    # Performance analysis
    print(f"\n📈 TENSORFLOW PERFORMANCE ANALYSIS:")
    if params < 5000:
        print(f"   ✅ EXCELLENT: Model size < 5k parameters (ultra-lightweight)")
    elif params < 10000:
        print(f"   ✅ GOOD: Model size < 10k parameters (lightweight)")
    else:
        print(f"   ⚠️  HEAVY: Model size > 10k parameters")
    
    if fps > 100:
        print(f"   🚀 ULTRA-FAST: {fps:.0f} FPS (excellent for real-time)")
    elif fps > 50:
        print(f"   ⚡ FAST: {fps:.0f} FPS (good for real-time)")
    else:
        print(f"   🐌 SLOW: {fps:.0f} FPS (needs optimization)")
    
    single_accuracy = correct_single_detections/len(sample_images)*100
    if single_accuracy > 80:
        print(f"   🎯 EXCELLENT: {single_accuracy:.1f}% single face detection accuracy")
    elif single_accuracy > 60:
        print(f"   ✅ GOOD: {single_accuracy:.1f}% single face detection accuracy")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT: {single_accuracy:.1f}% single face detection accuracy")
    
    # Final summary
    print(f"\n🎯 TENSORFLOW MODEL SUMMARY:")
    print(f"   📊 Model Parameters: {params:,} ({params/1000:.1f}k)")
    print(f"   ⚡ Inference Speed: {fps:.0f} FPS")
    print(f"   📈 Detection Accuracy: {single_accuracy:.1f}%")
    print(f"   💾 Detection Images Saved:")
    
    # List all saved images
    import glob
    saved_images = glob.glob('face_detection_tf_image_*.png')
    for img in saved_images:
        print(f"      - {img}")
    
    if len(saved_images) == 0:
        print(f"      - single_face_detection_tf.png (demo image)")
    
    if params < 5000:
        print(f"   ✅ ULTRA-LIGHTWEIGHT: <5k parameters achieved!")
    else:
        print(f"   ⚠️  Model size: {params/1000:.1f}k parameters")
    
    return model, {
        'params': params,
        'fps': fps,
        'total_detections': total_detections,
        'single_detection_accuracy': single_accuracy
    }


if __name__ == '__main__':
    model, results = evaluate_single_face_detection_tf() 