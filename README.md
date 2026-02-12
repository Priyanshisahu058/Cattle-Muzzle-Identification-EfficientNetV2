📌 Overview
This project implements an automated, non-invasive system for individual cattle identification using muzzle patterns—often referred to as "Bovine Fingerprints". Unlike traditional ear-tagging, which is invasive and prone to loss, this system leverages EfficientNetV2 to provide a secure, permanent biometric solution.

🚀 Key Performance & Results
Architecture: EfficientNetV2-S (6.8x smaller and faster than previous SOTA models).
Dataset: 4,923 high-resolution images across 268 unique subjects.
Inference Accuracy: Successfully achieved an 83.83% Confidence Level in identifying individual cattle.
Optimization: CPU-optimized training utilizing Transfer Learning from ImageNet weights.
🧠 Deep Learning vs. Classical ML
This project marks a significant leap from traditional "handcrafted" feature extraction.
Classical ML (HOG/SVM): Requires manual Region of Interest (ROI) extraction and expert feature engineering.
Our Approach (Deep Learning): Automatically learns complex textures and features directly from raw data, making the system inherently robust to lighting and angle variations.

🧬 Project Pipeline
Data Acquisition: Collection of 4,923 muzzle images.
Preprocessing: Standardization to 224x224 and RGB normalization.
Architecture: Deployment of EfficientNetV2-S for parameter economy.
Transfer Learning: Reconfigured the final layer to specialized muzzle recognition for 268 IDs.
Validation: Inference testing showing correct classification (e.g., Subject ID: 0

📂 Project Structure
CATTLE ID/
├── data/
│   └── sample_images/ (5-10 images for testing)
├── models/
│   └── (Download weights from Google Drive link in README)
├── scripts/
│   ├── data_loader.py
│   ├── model_setup.py
│   ├── predict.py
│   └── train.py
├── Biometric_Cattle_ID_Report.pdf
└── README.md

🔧 Installation & Usage
Clone the Repo:
git clone https://github.com/Priyanshisahu058/Cattle-Muzzle-Identification-EfficientNetV2.git
Run Prediction:
py scripts/predict.py

🔮 Future Scope
Edge Deployment: Optimizing the pipeline for real-time farm applications on mobile/low-resource devices.
Generalization: Evaluating reliability across different breeds and environmental conditions.
