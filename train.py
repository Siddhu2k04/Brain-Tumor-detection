import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORMS (AUGMENTATION) =================
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
    batch_size=64,
    shuffle=True,
    num_workers=0
)

print("Classes:", train_data.classes)

# ================= MODEL =================
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 4)

model = model.to(device)

# ================= LOSS & OPTIMIZER =================
criterion = nn.CrossEntropyLoss()

# Train only final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ================= TRAINING =================
epochs = 12

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

# ================= SAVE MODEL =================
torch.save(model.state_dict(), "brain_model.pth")

print("\n🔥 Model trained successfully (High Accuracy Version)")