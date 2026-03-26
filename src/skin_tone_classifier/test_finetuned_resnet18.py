import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "skin_tone_classifier_finetuned_resnet18_model.pth")
face_cascade_path = os.path.abspath(
    os.path.join(current_dir, "..", "..", "open_cv_models", "haarcascade_frontalface_default.xml")
)

num_classes = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['Black', 'Brown', 'White']

model = models.resnet18(weights = None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    raise FileNotFoundError(f"Unable to load Haar cascade file from: {face_cascade_path}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_skin_tone(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image from: {image_path}")

    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        raise ValueError("No face detected in the input image.")

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    face_bgr = image_bgr[y:y + h, x:x + w]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(face_rgb)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]


if __name__ == "__main__":
    prediction = predict_skin_tone("../../dataset/raw_dataset/skin_tone_testset/whiteblack.jpg")
    print(f"Predicted skin tone: {prediction}")
