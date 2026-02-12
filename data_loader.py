import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders():
    # 1. Transform: Resize and convert to numbers (Tensors)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load: Tell it where your folders are
    train_data = datasets.ImageFolder("data/processed/train", transform=transform)
    val_data = datasets.ImageFolder("data/processed/val", transform=transform)

    # 3. Batch: Look at 8 images at a time so the laptop doesn't crash
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=2)

    return train_loader, val_loader, len(train_data.classes)

if __name__ == "__main__":
    _, _, num_cows = get_loaders()
    print(f"✅ Data Loader Ready! Identifying {num_cows} cows.")