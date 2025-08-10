import torch
import os
import time
from PIL import Image, ImageDraw
from torchvision import transforms
from ultra_light_model import MicroFaceDetector, count_parameters


def load_trained_model(model_path='full_dataset_model.pth'):
    """Load the trained model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MicroFaceDetector().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded trained model from {model_path}")
        return model
    except:
        print(f"⚠️  Could not load {model_path}, using untrained model")
        return model


def detect_single_face(model, image_path, conf_threshold=0.5):
    """Detect exactly 1 face per image"""
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Get center prediction (most important)
    center_pred = torch.sigmoid(predictions[0, :, 3, 3])  # Center of 7x7 grid
    
    # Check if confidence is high enough
    if center_pred[4] > conf_threshold:
        # Extract bounding box from center prediction
        cx, cy, w, h = center_pred[:4].tolist()
        
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
            'confidence': center_pred[4].item(),
            'center': [cx, cy]
        }
        return image, [detection]  # Return as list for consistency
    else:
        return image, []  # No face detected


def evaluate_single_face_detection():
    """Evaluate single face detection performance"""
    
    print("🔍 SINGLE FACE DETECTION EVALUATION")
    print("=" * 50)
    
    # Load model
    model = load_trained_model()
    model.eval()
    
    # Model info
    params = count_parameters(model)
    print(f"📊 Model Parameters: {params:,} ({params/1000:.1f}k)")
    
    # Speed test
    device = next(model.parameters()).device
    test_input = torch.randn(1, 3, 112, 112).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(test_input)
    
    # Speed test
    start_time = time.time()
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(test_input)
    
    total_time = time.time() - start_time
    fps = num_runs / total_time
    latency = total_time / num_runs * 1000
    
    print(f"⚡ Speed: {fps:.0f} FPS | {latency:.1f}ms latency")
    
    # Test on sample images
    print(f"\n📸 TESTING ON SAMPLE IMAGES:")
    
    data_dir = 'data/lfw-deepfunneled/lfw-deepfunneled'
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
            image, detections = detect_single_face(model, image_path, conf_threshold=0.5)
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
                
                # Draw detection on first image
                if i == 0:
                    draw_image = image.copy()
                    draw = ImageDraw.Draw(draw_image)
                    
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    
                    # Draw rectangle
                    draw.rectangle(bbox, outline='red', width=3)
                    
                    # Add confidence text
                    draw.text((bbox[0], bbox[1]-20), f'{confidence:.2f}', fill='red')
                    
                    # Save the image
                    draw_image.save('single_face_detection.png')
                    print(f"   💾 Saved detection visualization to: single_face_detection.png")
            
            elif num_faces == 0:
                print(f"   ⚠️  No face detected")
            else:
                print(f"   ❌ Multiple faces detected: {num_faces}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print(f"\n🎯 EVALUATION SUMMARY:")
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
    print(f"\n📈 PERFORMANCE ANALYSIS:")
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
    
    return model, {
        'params': params,
        'fps': fps,
        'total_detections': total_detections,
        'single_detection_accuracy': single_accuracy
    }


if __name__ == '__main__':
    model, results = evaluate_single_face_detection() 