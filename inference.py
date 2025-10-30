import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def load_model(model_path, num_classes, device):
    model = models.resnet18(weights=None)  
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, transform, class_names, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 5, dim=1)
    
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # Get class names and probabilities
    top_predictions = []
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        top_predictions.append({
            'rank': i + 1,
            'class': class_names[idx],
            'probability': prob,
            'percentage': prob * 100
        })
    
    return top_predictions

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    MODEL_PATH = "final_resnet18_license_plates_10_30_2025.pth"
    NUM_CLASSES = 51
    
    CLASS_NAMES = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    ]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found!")
        return
    
    model = load_model(MODEL_PATH, NUM_CLASSES, device)
    
    folder_path = "./manual_test_images"
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found!")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in os.listdir(folder_path) 
                  if os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"Processing {len(image_files)} images:")
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        predictions = predict_image(model, image_path, transform, CLASS_NAMES, device)
        print(image_file)
        for prediction in predictions:
            print(prediction.get("rank"), " -> ", prediction.get("class"), " ----- ", round(prediction.get("percentage"), 2))
        print()

if __name__ == "__main__":
    main()