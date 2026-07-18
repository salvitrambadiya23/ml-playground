import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("flower_model.pth", map_location=device))
model = model.to(device)
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    print(f"Prediction: {class_names[predicted.item()]} ({confidence.item()*100:.2f}% confidence)")

if __name__ == "__main__":
    predict(sys.argv[1])