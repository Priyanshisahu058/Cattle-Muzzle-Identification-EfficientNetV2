import splitfolders
import os

# 1. Path to your raw images
input_folder = "data/raw/BeefCattle_Muzzle_Individualized"
output_folder = "data/processed"

# DIAGNOSTIC CHECKS
print(f"--- Checking Directories ---")
if not os.path.exists(input_folder):
    print(f"❌ ERROR: The path '{input_folder}' does not exist.")
    print("Double check if your folder is named exactly 'BeefCattle_Muzzle_Individualized'.")
else:
    # Check if there are actually folders inside
    subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    print(f"✅ Found input folder.")
    print(f"📂 Number of cow folders found: {len(subfolders)}")

    if len(subfolders) == 0:
        print("❌ ERROR: The folder is empty or has no subfolders. splitfolders needs subfolders (like cattle_0100) to work.")
    else:
        print(f"--- Starting Split ---")
        # Start the split
        splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, .2), move=False)
        print(f"✅ DONE! Refresh your VS Code sidebar and check the '{output_folder}' folder.")
