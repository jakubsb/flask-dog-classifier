import torch
from torch.nn.functional import softmax
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
import os
from definitions import ROOT_DIR

def preprocess_image(image_path, device=torch.device("cpu")):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    return input_tensor


def predict_class(input_tensor, device=torch.device("cpu")):
    class_names = load_labels()
    
    model = load_model()
    model.eval()

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)

    probs = softmax(output, dim=1)
    p, predicted_idx = torch.max(probs, 1)
    predicted_class_idx = predicted_idx.item()

    predicted_class = class_names[predicted_class_idx]

    return predicted_class, p.item()

def load_labels():
    
    with open(os.path.join(ROOT_DIR, 'labels/labels.txt'), 'r') as f:
        breeds = f.read()
        labels = breeds.split('\n')

    return labels

def load_model():
    class CustomClassifier(nn.Module):
        def __init__(self, num_classes):
            super(CustomClassifier, self).__init__()
            self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_features, num_classes)

        def forward(self, x):
            x = self.resnet(x)
            return x

    model = CustomClassifier(120)
    device = torch.device("cpu")
    model = model.to(device)

    checkpoint = torch.load(os.path.join(ROOT_DIR, 'model/model_dict.pth'), map_location='cpu')
    model.load_state_dict(checkpoint)

    model.to(device)

    return model
