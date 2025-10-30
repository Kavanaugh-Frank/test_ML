import torch
import torchvision
import numpy as np
from PIL import Image

print("=== GPU Diagnostics ===")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test GPU tensor operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x @ y  # Matrix multiplication on GPU
    print("GPU computation test: SUCCESS")
else:
    print("Running on CPU")

print(f"NumPy version: {np.__version__}")
print(f"Pillow version: {Image.__version__}")