# FaceID Detection with PyTorch (Edge Optimized)

This project implements a face recognition pipeline using PyTorch, trained on the open-source LFW dataset. The model is optimized for edge inference using MobileNetV2 and can be exported to ONNX or TorchScript for deployment.

## Features
- Lightweight MobileNetV2 backbone for edge devices
- Trains on LFW (Labeled Faces in the Wild) dataset
- Exports to ONNX/TorchScript for fast inference
- Simple inference script for single image

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (for my local venv): Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process & venv_virtualenv\Scripts\Activate.ps1
2. Download and prepare the dataset (handled automatically by the code).

## Training
```bash
python train.py
```

## Export for Edge Inference
```bash
python export.py
```

## Run Inference
```bash
python infer.py --image path_to_image.jpg
```

--- 