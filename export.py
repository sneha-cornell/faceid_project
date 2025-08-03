import torch
from model import load_model
from dataset import get_lfw_dataloaders


def export_models(weights_path='best_model.pth', img_size=96, device='cpu'):
    _, _, classes = get_lfw_dataloaders(img_size=img_size)
    num_classes = len(classes)
    model = load_model(num_classes, weights_path=weights_path, device=device)
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    # Export to TorchScript
    traced = torch.jit.trace(model, dummy_input)
    traced.save('model_scripted.pt')
    print('Saved TorchScript model to model_scripted.pt')
    # Export to ONNX
    torch.onnx.export(model, dummy_input, 'model.onnx', input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      opset_version=11)
    print('Saved ONNX model to model.onnx')

if __name__ == '__main__':
    export_models() 