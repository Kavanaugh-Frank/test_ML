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
from collections import Counter

def main():
    # -------------------------------
    # 1. Dataset directories
    # -------------------------------
    data_dir = "./plates"
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "valid")
    test_dir  = os.path.join(data_dir, "test")

    # Check if directories exist
    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} not found. Please check your data structure.")

    # Automatically determine number of classes
    num_classes = len(os.listdir(train_dir))
    print(f"Number of classes detected: {num_classes}")

    # -------------------------------
    # 2. Device
    # -------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------------
    # 3. Enhanced Data transforms
    # -------------------------------
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
        x: DataLoader(datasets_dict[x], batch_size=32, shuffle=(x=='train'), num_workers=4)  # Set to 0 for compatibility
        for x in ['train','valid','test']
    }

    dataset_sizes = {x: len(datasets_dict[x]) for x in ['train','valid','test']}
    class_names = datasets_dict['train'].classes
    print("Classes:", class_names)
    print("Dataset sizes:", dataset_sizes)

    # -------------------------------
    # 5. Check class balance and compute weights
    # -------------------------------
    train_labels = [label for _, label in datasets_dict['train']]
    class_counts = Counter(train_labels)
    print("Class distribution:", class_counts)
    
    # Compute class weights for imbalanced data
    class_weights = 1.0 / torch.tensor([class_counts[i] for i in range(num_classes)], dtype=torch.float32)
    class_weights = class_weights.to(device)
    print("Class weights:", class_weights)

    # -------------------------------
    # 6. Enhanced Model setup
    # -------------------------------
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Unfreeze last layers instead of all
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Enhanced classifier head
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
    # 7. Enhanced Loss and Optimizer
    # -------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    
    # Fixed scheduler - removed verbose parameter
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    # -------------------------------
    # 8. Enhanced Training function
    # -------------------------------
    def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=50):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = float('inf')
        train_losses, val_losses = [], []
        early_stopping_patience = 7
        no_improve = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*20}')
            
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                if phase == 'train':
                    train_losses.append(epoch_loss)
                else:
                    val_losses.append(epoch_loss)
                    # Step scheduler based on validation loss
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(epoch_loss)
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        print(f"Learning rate reduced from {old_lr} to {new_lr}")

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Early stopping check
                if phase == 'valid':
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        # Save best model
                        torch.save(model.state_dict(), 'best_resnet18_license_plates.pth')
                        print(f"New best model saved with accuracy: {best_acc:.4f}")

            print(f'Early stopping counter: {no_improve}/{early_stopping_patience}')
            
            # Early stopping
            if no_improve >= early_stopping_patience:
                print("Early stopping triggered!")
                break
            print()

        print(f'Best val Acc: {best_acc:.4f}')
        model.load_state_dict(best_model_wts)
        return model, train_losses, val_losses

    # -------------------------------
    # 9. Train the model
    # -------------------------------
    print("Starting training...")
    model, train_losses, val_losses = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=50
    )

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
    
    # Detailed results
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Per-class accuracy
        cm = confusion_matrix(all_labels, all_preds)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print("\nPer-class accuracy:")
        for i, acc in enumerate(per_class_accuracy):
            print(f"{class_names[i]}: {acc:.4f}")
    except ImportError:
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("Install scikit-learn for detailed classification report")

    # -------------------------------
    # 12. Print training summary
    # -------------------------------
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Device used: {device}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {dataset_sizes['train']}")
    print(f"Validation samples: {dataset_sizes['valid']}")
    print(f"Test samples: {dataset_sizes['test']}")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("Best model saved as: 'best_resnet18_license_plates.pth'")
    print("Final model saved as: 'final_resnet18_license_plates.pth'")

# -------------------------------
# Run main with error handling
# -------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please ensure:")
        print("1. Your data directory structure is: ./plates/train/, ./plates/valid/, ./plates/test/")
        print("2. Each subdirectory contains folders named by state (e.g., CA, NY, TX, etc.)")
        print("3. You have installed all required packages: torch, torchvision")
        raise