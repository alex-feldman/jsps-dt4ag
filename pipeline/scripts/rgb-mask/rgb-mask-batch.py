# simple script for creating masked images from images and pre-created corresponding masks
# run this script in the root folder containing 'images/' and 'masks/' dirs

import os
from pathlib import Path
from PIL import Image

# Define root paths
base_dir = Path("images")
mask_root = base_dir / "masks"
output_dir = Path("masked-images")  # Recommended to save to a new folder

print('b',base_dir)
print('m',mask_root)
print('o',output_dir)

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Iterate through all images, excluding the masks folder
for img_path in base_dir.rglob("*"):
    # Skip directories and files inside the masks root
    if img_path.is_dir() or mask_root in img_path.parents:
        continue
    
    # Supported image extensions
    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue

    # Determine the relative path to maintain structure (e.g., 'camera-1/image-1.jpg')
    relative_path = img_path.relative_to(base_dir)
    print('rp',relative_path)
    
    # Construct corresponding mask path
    mask_path = mask_root / relative_path
    print('mp',mask_path)
    # for now, hardcode rename mask extension to .png
    mask_path = (mask_path.with_suffix('.png'))
    print('mp',mask_path)

    if mask_path.exists():
        # Open main image and mask
        img = Image.open(img_path).convert("RGBA")  # Ensure RGBA for transparency
        mask = Image.open(mask_path).convert("L")   # Convert mask to grayscale (L)
        
        # Ensure they are the same size before applying
        if img.size == mask.size:
            img.putalpha(mask)  # Add mask as alpha channel
            print('same size ok')
            
            # Prepare output save path
            save_path = output_dir / relative_path
            save_path = save_path.with_suffix('.png') # Alpha requires PNG
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            img.save(save_path)
            print(f"Processed: {relative_path}")
        else:
            print(f"Size mismatch for {relative_path}: {img.size} vs {mask.size}")
    else:
        print(f"Mask missing for: {relative_path}")

