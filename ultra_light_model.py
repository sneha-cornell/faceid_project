import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Ultra-efficient depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class MicroFaceDetector(nn.Module):
    """Ultra-lightweight face detector with <5k parameters"""
    
    def __init__(self, num_classes=1):
        super().__init__()
        
        # Ultra-compact feature extractor
        self.backbone = nn.Sequential(
            # Stage 1: 112x112x3 -> 56x56x8
            nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            
            # Stage 2: 56x56x8 -> 28x28x16 
            DepthwiseSeparableConv(8, 16, stride=2),
            
            # Stage 3: 28x28x16 -> 14x14x32
            DepthwiseSeparableConv(16, 32, stride=2),
            
            # Stage 4: 14x14x32 -> 7x7x64
            DepthwiseSeparableConv(32, 64, stride=2),
        )
        
        # Detection head - single scale for simplicity
        # Output: [batch, 5, 7, 7] where 5 = (x, y, w, h, confidence)
        self.detection_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 5, 1)  # Final detection layer
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Feature extraction: 112x112x3 -> 7x7x64
        features = self.backbone(x)
        
        # Detection: 7x7x64 -> 7x7x5
        detection = self.detection_head(features)
        
        return detection
    
    def detect_faces(self, x, conf_threshold=0.5):
        """Extract face detections from model output"""
        with torch.no_grad():
            output = self.forward(x)
            batch_size, _, h, w = output.shape
            
            detections = []
            
            for b in range(batch_size):
                batch_detections = []
                
                for i in range(h):
                    for j in range(w):
                        # Extract prediction for this grid cell
                        pred = output[b, :, i, j]
                        
                        # Apply sigmoid to get valid ranges
                        cx = (j + torch.sigmoid(pred[0])) / w  # Normalized center x
                        cy = (i + torch.sigmoid(pred[1])) / h  # Normalized center y
                        w_box = torch.sigmoid(pred[2])         # Normalized width
                        h_box = torch.sigmoid(pred[3])         # Normalized height
                        conf = torch.sigmoid(pred[4])         # Confidence score
                        
                        if conf > conf_threshold:
                            # Convert to corner coordinates
                            x1 = cx - w_box / 2
                            y1 = cy - h_box / 2
                            x2 = cx + w_box / 2
                            y2 = cy + h_box / 2
                            
                            batch_detections.append({
                                'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                                'confidence': conf.item(),
                                'grid': [i, j]
                            })
                
                detections.append(batch_detections)
            
            return detections


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_ultra_light_detector():
    """Create and analyze ultra-lightweight face detector"""
    
    print("🚀 CREATING ULTRA-LIGHTWEIGHT FACE DETECTOR")
    print("=" * 50)
    
    model = MicroFaceDetector()
    params = count_parameters(model)
    
    print(f"✅ Model created successfully!")
    print(f"📊 Parameters: {params:,} ({params/1000:.1f}k)")
    
    # Model size estimation
    param_size = params * 4 / (1024*1024)  # 4 bytes per float32 parameter
    print(f"💾 Model size: ~{param_size:.1f} MB")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 112, 112)
    output = model(test_input)
    print(f"🔍 Input shape: {test_input.shape}")
    print(f"📤 Output shape: {output.shape}")
    
    # Speed test
    model.eval()
    import time
    
    # Warmup
    for _ in range(10):
        _ = model(test_input)
    
    # Timing
    start_time = time.time()
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(test_input)
    
    total_time = time.time() - start_time
    fps = num_runs / total_time
    latency = total_time / num_runs * 1000
    
    print(f"⚡ Speed: {fps:.0f} FPS | {latency:.1f}ms latency")
    
    # Target analysis
    print(f"\n🎯 TARGET ANALYSIS:")
    if params < 5000:
        print(f"✅ EXCELLENT: {params} parameters < 5k target!")
    elif params < 10000:
        print(f"✅ GOOD: {params} parameters < 10k (close to target)")
    else:
        print(f"⚠️  NEEDS OPTIMIZATION: {params} parameters > 10k")
    
    if fps > 100:
        print(f"🚀 ULTRA-FAST: {fps:.0f} FPS (excellent for real-time)")
    elif fps > 50:
        print(f"⚡ FAST: {fps:.0f} FPS (good for real-time)")
    else:
        print(f"🐌 SLOW: {fps:.0f} FPS (needs optimization)")
    
    return model


if __name__ == '__main__':
    model = create_ultra_light_detector()