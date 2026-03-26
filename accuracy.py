import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("brain_model.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load test data
test_data = datasets.ImageFolder("data/train", transform=transform)
test_loader = DataLoader(test_data, batch_size=32)

correct = 0
total = 0

# Testing loop
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

# Accuracy
accuracy = (correct / total) * 100
print("Accuracy:", accuracy, "%")