import torch
from torchvision import transforms
from PIL import Image
import argparse
from dataset import get_lfw_dataloaders
from facenet_pytorch import MTCNN
import cv2
import numpy as np


def load_classes(img_size=96):
    _, _, classes = get_lfw_dataloaders(img_size=img_size)
    return classes

def preprocess_face(face_img, img_size=96):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(face_img).unsqueeze(0)

def detect_and_recognize(image_path, model_path='model_scripted.pt', img_size=96, device='cpu', save_path=None):
    # Load classes and model
    classes = load_classes(img_size)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Detect faces
    mtcnn = MTCNN(keep_all=True, device=device)
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        print('No faces detected.')
        return
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        face = img.crop((x1, y1, x2, y2))
        input_tensor = preprocess_face(face, img_size=img_size).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            top1_prob, top1_idx = torch.max(prob, 1)
            name = classes[top1_idx.item()]
            label = f'{name} ({top1_prob.item():.2f})'
        # Draw bounding box and label
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_cv, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    # Show and/or save
    cv2.imshow('Face Recognition', img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if save_path:
        cv2.imwrite(save_path, img_cv)
        print(f'Saved result to {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='model_scripted.pt', help='Path to TorchScript model')
    parser.add_argument('--img_size', type=int, default=96, help='Image size (default: 96)')
    parser.add_argument('--save', type=str, default=None, help='Path to save output image (optional)')
    args = parser.parse_args()
    detect_and_recognize(args.image, args.model, args.img_size, args.save) 