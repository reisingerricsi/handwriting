
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Pretrained model from Hugging Face
MODEL_NAME = "microsoft/trocr-base-handwritten"

# Input image containing handwritten text
IMAGE_PATH = Path("handwriting.png")

# Select GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the pretrained TrOCR model and processor."""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {DEVICE}")

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    return processor, model


def recognize_text(image_path, processor, model):
    """Recognize handwritten text from an image."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path.resolve()}"
        )

    # TrOCR expects an image of a text line
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    # Run inference without calculating gradients
    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128
        )

    # Convert generated tokens into readable text
    recognized_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return recognized_text


def main():
    processor, model = load_model()

    try:
        text = recognize_text(
            IMAGE_PATH,
            processor,
            model
        )

        print("\nRecognized text:")
        print("-" * 40)
        print(text)
        print("-" * 40)

    except FileNotFoundError as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
