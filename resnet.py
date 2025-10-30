import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from collections import Counter
import numpy as np
from pathlib import Path
from PIL import Image
import json

class ImageFolderFromManifest(Dataset):
    def __init__(self, img_dir: str, manifest_path: str, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform

        # Read JSONL manifest line by line
        self.data = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

        self.classes = sorted(list({item["class-label"] for item in self.data}))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]["source-ref"]
        label_name = self.data[idx]["class-label"]
        label = self.class_to_idx[label_name]

        # this will have to change when the images are in S3
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # -------------------------------
    # 1. Manifests paths
    # -------------------------------
    data_dir = Path("./flat_plates")
    train_manifest = data_dir / "train" / "train_manifest.jsonl"
    val_manifest   = data_dir / "valid" / "valid_manifest.jsonl"
    test_manifest  = data_dir / "test" / "test_manifest.jsonl"

    # Check manifest files exist
    for manifest in [train_manifest, val_manifest, test_manifest]:
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest {manifest} not found. Please check your dataset.")

    # -------------------------------
    # 2. Device
    # -------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------------
    # 3. Data transforms
    # -------------------------------
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # -------------------------------
    # 4. Datasets and dataloaders
    # -------------------------------
    datasets_dict = {
        'train': ImageFolderFromManifest("./flat_plates/train", train_manifest, transform=data_transforms['train']),
        'valid': ImageFolderFromManifest("./flat_plates/valid", val_manifest, transform=data_transforms['valid']),
        'test':  ImageFolderFromManifest("./flat_plates/test", test_manifest, transform=data_transforms['test']),
    }

    dataloaders = {
        x: DataLoader(datasets_dict[x], batch_size=32, shuffle=(x=='train'), num_workers=4)
        for x in ['train','valid','test']
    }

    dataset_sizes = {x: len(datasets_dict[x]) for x in ['train','valid','test']}
    class_names = datasets_dict['train'].classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Dataset sizes:", dataset_sizes)

    # -------------------------------
    # 5. Class weights
    # -------------------------------
    train_labels = [label for _, label in datasets_dict['train']]
    class_counts = Counter(train_labels)
    class_weights = 1.0 / torch.tensor([class_counts[i] for i in range(num_classes)], dtype=torch.float32)
    class_weights = class_weights.to(device)
    print("Class weights:", class_weights)

    # -------------------------------
    # 6. Model setup
    # -------------------------------
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Unfreeze last layers
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Custom classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    # -------------------------------
    # 7. Loss, optimizer, scheduler
    # -------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=0.001, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # -------------------------------
    # 8. Training function
    # -------------------------------
    def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=50):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = float('inf')
        early_stopping_patience = 7
        no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*20}')
            for phase in ['train', 'valid']:
                model.train() if phase=='train' else model.eval()
                running_loss, running_corrects = 0.0, 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase=='train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase=='train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase=='train':
                    train_losses.append(epoch_loss)
                else:
                    val_losses.append(epoch_loss)
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(epoch_loss)
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        print(f"LR reduced: {old_lr} -> {new_lr}")

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Early stopping & best model
                if phase=='valid':
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), 'best_resnet18_license_plates.pth')
                        print(f"New best model saved: Acc={best_acc:.4f}")

            print(f'Early stopping counter: {no_improve}/{early_stopping_patience}')
            if no_improve >= early_stopping_patience:
                print("Early stopping triggered!")
                break
            print()

        model.load_state_dict(best_model_wts)
        print(f'Best val Acc: {best_acc:.4f}')
        return model, train_losses, val_losses

    # -------------------------------
    # 9. Train
    # -------------------------------
    print("Starting training...")
    model, train_losses, val_losses = train_model(model, dataloaders, dataset_sizes,
                                                  criterion, optimizer, scheduler, device,
                                                  num_epochs=50)

    # -------------------------------
    # 10. Save final model
    # -------------------------------
    torch.save(model.state_dict(), 'final_resnet18_license_plates.pth')
    print("Final model saved as 'final_resnet18_license_plates.pth'")

    # -------------------------------
    # 11. Test evaluation
    # -------------------------------
    print("\nEvaluating on test set...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
