import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

classes = ['PURPLE_CHLORIS', 'CROWFOOT_GRASS', 'CELOSIA_ARGENTEA_L']

device = torch.device("cpu")

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 3)
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return classes[pred.item()]

# Example
print(predict("sample.jpg"))