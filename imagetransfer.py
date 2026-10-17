
import os
import shutil

# Source and destination folders
source_folder = r"dataset/dataset1/img_align_celeba"
destination_folder = r"dataset_20_image/img_align_celeba"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Get image files from source folder
images = [
    file for file in os.listdir(source_folder)
    if file.lower().endswith(image_extensions)
]

# Sort images for consistent ordering
images.sort()

# Copy first 20 images
for image in images[:500]:
    source_path = os.path.join(source_folder, image)
    destination_path = os.path.join(destination_folder, image)

    shutil.copy2(source_path, destination_path)
    print(f"Copied: {image}")

print(f"\nDone! Copied {min(20, len(images))} images.")
