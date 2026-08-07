import torch
from PIL import Image
import matplotlib.pyplot as plt
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load SAM3 model
#model = build_sam3_image_model()
model = build_sam3_image_model(device='cpu')
#processor = Sam3Processor(model)
processor = Sam3Processor(model, device='cpu')

# Load image
image = Image.open("image.jpg")
inference_state = processor.set_image(image)

model = model.to('cpu') 
processor = processor.to('cpu')

# Here's the magic moment! Use text to describe what you want
text_prompt = "plant"  # You can input any concept
output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

# Get segmentation results
masks = output["masks"]        # Segmentation masks
boxes = output["boxes"]        # Bounding boxes
scores = output["scores"]      # Confidence scores

# Display results
def show_results(image, masks, boxes, scores, text_prompt):
    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(15, 5))
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Show segmentation results
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        axes[i+1].imshow(image)
        axes[i+1].imshow(mask, alpha=0.6, cmap='viridis')
        
        # Draw bounding box
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color='red', linewidth=2)
        axes[i+1].add_patch(rect)
        
        axes[i+1].set_title(f"Score: {score:.3f}")
        axes[i+1].axis('off')
    
    plt.suptitle(f'Text Prompt: "{text_prompt}"', fontsize=16)
    plt.tight_layout()
    plt.show()

show_results(image, masks, boxes, scores, text_prompt)
