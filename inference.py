import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

class LicensePlateResNet18(nn.Module):
    def __init__(self, num_classes):
        super(LicensePlateResNet18, self).__init__()
        # directly use resnet18, no 'self.model'
        self.convnet = models.resnet18(weights=None)
        in_features = self.convnet.fc.in_features
        self.convnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.convnet(x)

def predict_image(model, image_path, class_names=None, device=None):
    """
    Runs inference on a single image using a trained PyTorch model.
    
    Args:
        model (torch.nn.Module): The trained model.
        image_path (str): Path to the input image.
        class_names (list[str], optional): List of class names for human-readable output.
        device (torch.device, optional): 'cuda' or 'cpu'. Auto-detects if not provided.
    
    Returns:
        dict: {
            'predicted_index': int,
            'predicted_class': str (if class_names provided),
            'probabilities': list[float]
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Put model in eval mode and move to device
    model.eval()
    model.to(device)

    # Define transforms (match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    top5_prob = top5_prob.cpu().numpy()
    top5_idx = top5_idx.cpu().numpy()

    # Prepare results
    result = {
        "predicted_index": top5_idx[0],
        "probabilities": probabilities.cpu().numpy().tolist(),
        "top5": [
            {
                "class": class_names[i] if class_names else str(i),
                "probability": round(float(p) * 100, 2)
            }
            for i, p in zip(top5_idx, top5_prob)
        ]
    }

    if class_names:
        result["predicted_class"] = class_names[result["predicted_index"]]

    return result

# Number of output classes (adjust this to your dataset)
NUM_CLASSES = 51  # Example: 50 U.S. states

# Create model and load trained weights
model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 51)
model.load_state_dict(torch.load('resnet18_us_license_plates.pth', map_location='cpu'))
model.eval()

result = predict_image(model, "3.jpg", class_names=['ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO', 'CONNECTICUT', 'DELAWARE', 'FLORIDA', 'GEORGIA', 'HAWAI', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND', 'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA', 'MISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA', 'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON', 'WEST VIRGINIA', 'WISCONSIN', 'WYOMING'])
for classification in result["top5"]:
    print(classification["class"], classification["probability"])