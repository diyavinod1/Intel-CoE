import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Try Intel optimization (optional)
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
    print("✅ Using Intel oneAPI (IPEX)")
except:
    IPEX_AVAILABLE = False
    print("⚠️ IPEX not available, running normally")

# Device
device = torch.device("cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Data
train_data = datasets.ImageFolder("dataset/train", transform=train_transform)
val_data = datasets.ImageFolder("dataset/val", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Model
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 3)
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

# Apply Intel optimization if available
if IPEX_AVAILABLE:
    model, optimizer = ipex.optimize(model, optimizer=optimizer)

best_acc = 0

# Training loop
for epoch in range(15):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, Val Accuracy={acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/best_model.pth")

print(f"🔥 Best Validation Accuracy: {best_acc:.2f}%")
