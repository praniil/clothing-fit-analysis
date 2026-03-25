import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import os
import cv2
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
fine_tuned_model_path = os.path.join(current_dir, "skin_tone_classifier_finetuned_resnet18_model.pth")

dataset_path = os.path.abspath(
    os.path.join(current_dir, "..", "..", "dataset", "raw_dataset", "skin_tone_classificaiton_dataset")
)
face_cascade_path = os.path.abspath(
    os.path.join(current_dir, "..", "..", "open_cv_models", "haarcascade_frontalface_default.xml")
)
 
#hyperparams
batch_size = 32
num_classes = 3
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

#transformation (resize images to 224 by 224 for ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], # mean
                        [0.229, 0.224, 0.225])  # standard deviation
])


class FaceCroppedImageFolder(datasets.ImageFolder):
    def __init__(self, root, face_cascade, transform=None):
        super().__init__(root=root, transform=transform)
        self.face_cascade = face_cascade
        self.face_samples = []

        for path, target in self.samples:
            image_bgr = cv2.imread(path)
            if image_bgr is None:
                continue

            gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                continue

            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            self.face_samples.append((path, target, (x, y, w, h)))

        if len(self.face_samples) == 0:
            raise ValueError("No faces were detected in the dataset. Please verify image quality and cascade settings.")

        print(f"Using {len(self.face_samples)} face-cropped samples out of {len(self.samples)} total images")

    def __len__(self):
        return len(self.face_samples)

    def __getitem__(self, index):
        path, target, (x, y, w, h) = self.face_samples[index]
        image_bgr = cv2.imread(path)
        face_bgr = image_bgr[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(face_rgb)

        if self.transform is not None:
            image = self.transform(image)

        return image, target


#loading the dataset
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise FileNotFoundError(f"Unable to load Haar cascade file from: {face_cascade_path}")

dataset = FaceCroppedImageFolder(dataset_path, face_cascade=face_cascade, transform=transform)
class_names = dataset.classes
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def tensor_to_bgr_image(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    image = image.clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def preview_face_detection(max_images=10):
    shown = 0
    for images, _ in train_loader:
        for image_tensor in images:
            image = tensor_to_bgr_image(image_tensor)
            cv2.imshow("face-cropped input", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            shown += 1
            if shown >= max_images:
                return

#loading the pretrained model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

#Replaing the last FC layer to 3 i.e. our classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model():
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), fine_tuned_model_path)
    print(f"Model saved to: {fine_tuned_model_path}")


if __name__ == "__main__":
    run_face_preview = False
    if run_face_preview:
        preview_face_detection()
    train_model()


