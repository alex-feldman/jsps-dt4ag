from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Load model with text encoder
model = build_efficientsam3_image_model(
    checkpoint_path="efficient_sam3_tinyvit_m_mobileclip_s1.pt",
    backbone_type="tinyvit",
    model_name="11m",
    text_encoder_type="MobileCLIP-S1"
)

image = Image.open("image.jpg")

# Process image and predict with text prompt
processor = Sam3Processor(model)
inference_state = processor.set_image(image)
inference_state = processor.set_text_prompt(inference_state, prompt="shoe")
masks, scores, _ = model.predict_inst(inference_state)
