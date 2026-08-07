"""Minimal EfficientSAM3 text-prompted segmentation, for checking a checkpoint runs.

Smaller and faster than the full SAM3 path in sl-sam3-qs-text.py. Prints mask
count and scores rather than plotting.

The prompt is REQUIRED and has no default: this pipeline is object-agnostic.
"""

import argparse

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True,
                   help="Path to the input image.")
    p.add_argument("--prompt", required=True,
                   help="Text description of what to segment, e.g. 'chair'.")
    p.add_argument("--checkpoint",
                   default="efficient_sam3_tinyvit_m_mobileclip_s1.pt",
                   help="Model checkpoint path (default: %(default)s).")
    p.add_argument("--backbone-type", default="tinyvit",
                   help="Backbone type (default: %(default)s).")
    p.add_argument("--model-name", default="11m",
                   help="Model size (default: %(default)s).")
    p.add_argument("--text-encoder-type", default="MobileCLIP-S1",
                   help="Text encoder (default: %(default)s).")
    return p.parse_args()


def main():
    args = parse_args()

    # Load model with text encoder
    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder_type,
    )

    image = Image.open(args.image)

    # Process image and predict with text prompt
    processor = Sam3Processor(model)
    inference_state = processor.set_image(image)
    inference_state = processor.set_text_prompt(inference_state, prompt=args.prompt)
    masks, scores, _ = model.predict_inst(inference_state)

    print(f'prompt: "{args.prompt}"  masks: {len(masks)}')
    for i, s in enumerate(scores):
        print(f"  {i}: score {float(s):.3f}")


if __name__ == "__main__":
    main()
