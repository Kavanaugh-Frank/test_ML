import torch
from torch.utils.data import Dataset
from PIL import Image
import pathlib
import json

class ImageFolderFromManifest(Dataset):
    def __init__(self, img_dir: str, manifest_path: str, transform=None):
        """
        Args:
            img_dir (str): Path to folder containing all images.
            manifest_path (str): Path to manifest file (CSV or JSON).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = pathlib.Path(img_dir)
        self.transform = transform
    
        with open(manifest_path, "r") as f:
            self.data = json.load(f)
        
        # Extract class names and map to indices
        self.classes = sorted(self.data["label"].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data[idx]["source-red"]
        label_name = self.data[idx]["class-label"]
        label = self.class_to_idx[label_name]

        # join the two paths
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
