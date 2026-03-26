import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= DATA =================
train_data = datasets.ImageFolder('data/train', transform=transform)

train_loader = DataLoader(
    train_data,
    batch_size=16,   # reduced for speed
    shuffle=True,
    num_workers=0
)

print("Classes:", train_data.classes)

# ================= MODEL =================
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block for better learning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 4)

model = model.to(device)

# ================= LOSS & OPTIMIZER =================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# ================= TRAINING =================
epochs = 12

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Batch logging
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(f"\n✅ Epoch [{epoch+1}/{epochs}] Completed")
    print(f"Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%\n")

# ================= SAVE MODEL =================
torch.save(model.state_dict(), "brain_model.pth")
torch.save(model, "brain_model_full.pth")

print("🔥 Model trained successfully!")
