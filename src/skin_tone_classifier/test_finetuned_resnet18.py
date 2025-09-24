import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "skin_tone_classifier_finetuned_resnet18_model.pth")

num_classes = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['Black', 'Brown', 'White']

model = models.resnet18(weights = None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_skin_tone(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]
