import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import copy
import numpy as np
from torchvision.models import ResNet18_Weights

def main():
    # -------------------------------
    # 1. Dataset directories
    # -------------------------------
    data_dir = "./plates"
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "valid")
    test_dir  = os.path.join(data_dir, "test")

    # Automatically determine number of classes
    num_classes = len(os.listdir(train_dir))
    print(f"Number of classes detected: {num_classes}")

    # -------------------------------
    # 2. Device
    # -------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------
    # 3. Data transforms
    # -------------------------------
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.ColorJitter(brightness=(0.3, 0.7)),  # darken
            # transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1,2.0)),  # blur
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.ColorJitter(brightness=(0.3, 0.7)),  # darken
            # transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1,2.0)),  # blur
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.ColorJitter(brightness=(0.3, 0.7)),  # darken
            # transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1,2.0)),  # blur
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # -------------------------------
    # 4. Datasets and dataloaders
    # -------------------------------
    datasets_dict = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'valid': datasets.ImageFolder(val_dir, data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test'])
    }

    dataloaders = {
        x: DataLoader(datasets_dict[x], batch_size=32, shuffle=(x=='train'), num_workers=4)  # use 0 first
        for x in ['train','valid','test']
    }

    dataset_sizes = {x: len(datasets_dict[x]) for x in ['train','valid','test']}
    class_names = datasets_dict['train'].classes
    print("Classes:", class_names)

    # -------------------------------
    # 5. Model setup
    # -------------------------------
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # -------------------------------
    # 6. Training function
    # -------------------------------
    def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=20):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*20}')
            for phase in ['train','valid']:
                model.train() if phase=='train' else model.eval()
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase=='train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase=='train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds==labels.data)

                if phase=='train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase=='valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            print()

        print(f'Best val Acc: {best_acc:.4f}')
        model.load_state_dict(best_model_wts)
        return model

    # -------------------------------
    # 7. Train
    # -------------------------------
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=20)

    # -------------------------------
    # 8. Save model
    # -------------------------------
    torch.save(model.state_dict(), 'resnet18_us_license_plates.pth')
    print("Model saved as 'resnet18_us_license_plates.pth'")

    # -------------------------------
    # 9. Test evaluation
    # -------------------------------
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_preds==all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

# -------------------------------
# 10. Run main
# -------------------------------
if __name__ == "__main__":
    main()
