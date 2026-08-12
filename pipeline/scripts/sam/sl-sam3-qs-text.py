"""Segment an image with SAM3 from a text prompt, and show the result.

A demo/inspection script, not part of the batch pipeline: it visualizes what a
given text prompt selects so you can choose one before running a batch.

The prompt is REQUIRED and has no default. SAM3 will segment whatever concept
you name, and this pipeline is object-agnostic, so there is no sensible default
to guess at.
"""

import argparse

import torch
from PIL import Image
import matplotlib.pyplot as plt
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True,
                   help="Path to the input image.")
    p.add_argument("--prompt", required=True,
                   help="Text description of what to segment, e.g. 'chair'. "
                        "Any concept SAM3 understands.")
    p.add_argument("--device", default="cpu",
                   help="Torch device (default: cpu). Use 'cuda' if a GPU is available.")
    p.add_argument("--save",
                   help="Write the figure to this path instead of opening a window. "
                        "Use this when running headless.")
    return p.parse_args()


def show_results(image, masks, boxes, scores, text_prompt, save_path=None):
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
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Wrote {save_path}")
    else:
        plt.show()


def main():
    args = parse_args()

    # Load SAM3 model
    model = build_sam3_image_model(device=args.device)
    processor = Sam3Processor(model, device=args.device)

    # Load image
    image = Image.open(args.image)
    inference_state = processor.set_image(image)

    model = model.to(args.device)
    processor = processor.to(args.device)

    # Use text to describe what you want
    output = processor.set_text_prompt(state=inference_state, prompt=args.prompt)

    # Get segmentation results
    masks = output["masks"]        # Segmentation masks
    boxes = output["boxes"]        # Bounding boxes
    scores = output["scores"]      # Confidence scores

    show_results(image, masks, boxes, scores, args.prompt, args.save)


if __name__ == "__main__":
    main()
