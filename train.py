import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_loaders
from model_setup import get_model

def start_training():
    device = torch.device("cpu")
    
    print("⏳ Step 1: Loading your cattle dataset... (This can take 1 minute)")
    train_loader, val_loader, num_cows = get_loaders()
    
    print(f"⏳ Step 2: Preparing the AI brain for {num_cows} cows...")
    model = get_model(num_cows).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    print("🚀 Step 3: Starting Training Loop!")
    for epoch in range(5):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/5")
        
        for images, labels in pbar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(model.state_dict(), "models/trained_cow_model.pth")
    print("🎉 Finished! Your AI is now trained.")

if __name__ == "__main__":
    start_training()