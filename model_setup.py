import torch
import torch.nn as nn
import timm
import os

def get_model(num_classes):
    # 1. Build the AI body (EfficientNetV2-S)
    model = timm.create_model('efficientnetv2_s', pretrained=False)
    
    # 2. Point to the file you just moved!
    weight_path = "models/efficientnet_weights.pth"
    
    if os.path.exists(weight_path):
        # 3. Load the brain offline
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        print("✅ SUCCESS: Brain loaded offline. Day 2 Complete!")
    else:
        print(f"❌ STILL MISSING: Please move the .pth file to {weight_path}")
        return None

    # 4. Rewire for your 268 cows
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    return model

if __name__ == "__main__":
    get_model(268)
    # --- ADD THIS TO THE VERY BOTTOM ---
if __name__ == "__main__":
    print("⏳ Finalizing setup...")
    model = get_model(268)
    if model:
        print("✅ SUCCESS! Day 2 is officially complete.")
        print("🚀 Your AI is now ready for Day 3 Training.")