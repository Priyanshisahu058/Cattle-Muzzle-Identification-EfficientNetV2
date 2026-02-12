import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model_setup import get_model
import os

def identify_cow(image_path):
    # 1. Basic Setup
    num_cows = 268 
    device = torch.device("cpu")
    
    # 2. Load the trained "knowledge" (the .pth file from Day 3)
    print("⏳ Loading trained model...")
    model = get_model(num_cows)
    
    # Path to your saved weights
    model_path = "models/trained_cow_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set to evaluation mode
        print("✅ Model loaded successfully.")
    else:
        print(f"❌ Error: Cannot find {model_path}. Did you finish training?")
        return

    # 3. Prepare the image (Transform it so the AI can read it)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Process the image
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)

    # 5. Make the Prediction
    print(f"🔍 Analyzing muzzle pattern in {image_path}...")
    with torch.no_grad():
        output = model(img_t)
        probabilities = F.softmax(output, dim=1)
        conf, pred = torch.max(probabilities, dim=1)

    # 6. Show the Final Result
    print("\n" + "="*30)
    print("🐄 AI PREDICTION RESULT")
    print("="*30)
    print(f"Predicted Cattle ID: {pred.item()}")
    print(f"Confidence Level:    {conf.item()*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    test_image = "test_cow.jpg" 
    
    if os.path.exists(test_image):
        print(f"🔍 AI is now analyzing {test_image}...")
        identify_cow(test_image)
    else:
        print(f"❌ Error: {test_image} not found in project folder!")