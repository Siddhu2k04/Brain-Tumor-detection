import torch
from torchvision import transforms, models
from PIL import Image
import os

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= CONFIG =================
MODEL_PATH = "brain_model.pth"
IMAGE_PATH = "test.jpg"

CLASS_NAMES = ["no_tumor", "glioma", "meningioma", "pituitary"]

# ================= LOAD MODEL =================
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 4)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= LOAD IMAGE =================
if not os.path.exists(IMAGE_PATH):
    print("❌ Image not found")
    exit()

try:
    img = Image.open(IMAGE_PATH)
    img = transform(img).unsqueeze(0).to(device)
except Exception as e:
    print("❌ Error:", e)
    exit()

# ================= PREDICTION =================
with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)

    confidence, pred = torch.max(probs, 1)

# ================= OUTPUT =================
print("\n===== RESULT =====")

pred_class = CLASS_NAMES[pred.item()]
confidence = confidence.item() * 100

print(f"Prediction  : {pred_class}")
print(f"Confidence  : {confidence:.2f}%")

print("\nClass probabilities:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"{cls}: {probs[0][i].item()*100:.2f}%")

print("\n⚠️ AI-assisted result. Not a medical diagnosis.")