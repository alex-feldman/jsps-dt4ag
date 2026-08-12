from PIL import Image

# Open the RGB image and the grayscale mask image
rgb_image = Image.open("image.jpg").convert("RGB")
mask_image = Image.open("image.jpg.png").convert("L") # Convert mask to grayscale ('L' mode)

# Ensure the RGB image has an alpha channel added
rgb_image.putalpha(mask_image)

# Save the resulting image with the embedded alpha channel
rgb_image.save("image_with_alpha.png")

