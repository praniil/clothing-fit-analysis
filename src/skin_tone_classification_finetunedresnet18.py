import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import os

fine_tuned_model_path = "skin_tone_classifier_finetuned_resnet18_model.pth"

if not os.path.exists(fine_tuned_model_path):
    with open(fine_tuned_model_path, 'w') as file:
        pass

else:
    print(f"the file {fine_tuned_model_path} already exists")
 
#hyperparams
batch_size = 16
num_classes = 3
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

#transformation (resize images to 224 by 224 for ResNet)
tranform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], # mean
                        [0.229, 0.224, 0.225])  # standard deviation
])


#loading the dataset
dataset = datasets.ImageFolder('../dataset/raw_dataset/skin_tone_classificaiton_dataset', transform=tranform)
print(len(dataset))
class_names = dataset.classes
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


#loading the pretrained model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

#Replaing the last FC layer to 3 i.e. our classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Training in loop
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
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")

#saving the model
torch.save(model.state_dict(), fine_tuned_model_path)


