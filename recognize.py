
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Pretrained model from Hugging Face
MODEL_NAME = "microsoft/trocr-base-handwritten"

# Input image containing handwritten text
DEFAULT_IMAGE = "handwriting.png"

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


def open_image(image_path):
    """Open an image, flattening any transparency onto white."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path.resolve()}"
        )

    image = Image.open(image_path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def split_lines(image, min_gap=8, min_height=12, margin=6):
    """Split a page image into text-line crops using row ink projection."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    # Ink is darker than the page background; Otsu-like split via mean of extremes
    low, high = np.percentile(gray, 2), np.percentile(gray, 98)
    if high - low < 30:
        return [image]
    ink = gray < (low + high) / 2

    # Rows containing meaningful ink (ignores specks and thin ruled lines)
    row_ink = ink.sum(axis=1) > max(2, 0.004 * ink.shape[1])

    segments, start, gap = [], None, 0
    for y, has_ink in enumerate(row_ink):
        if has_ink:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                segments.append((start, y - gap + 1))
                start, gap = None, 0
    if start is not None:
        segments.append((start, len(row_ink) - gap))

    crops = []
    for top, bottom in segments:
        if bottom - top < min_height:
            continue
        cols = np.where(ink[top:bottom].any(axis=0))[0]
        left, right = max(cols[0] - margin, 0), cols[-1] + margin
        crops.append(
            image.crop(
                (left, max(top - margin, 0), right, min(bottom + margin, image.height))
            )
        )
    return crops or [image]


def recognize_text(image, processor, model):
    """Recognize handwritten text from a single-line PIL image."""
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
    parser = argparse.ArgumentParser(description="Handwriting recognition with TrOCR")
    parser.add_argument("images", nargs="*", default=[DEFAULT_IMAGE])
    parser.add_argument(
        "--lines",
        action="store_true",
        help="split each image into text lines and recognize every line",
    )
    args = parser.parse_args()
    processor, model = load_model()

    for image_path in args.images:
        try:
            image = open_image(image_path)
        except FileNotFoundError as error:
            print(f"Error: {error}")
            continue

        crops = split_lines(image) if args.lines else [image]
        print(f"\n{image_path}:")
        for crop in crops:
            print(recognize_text(crop, processor, model))


if __name__ == "__main__":
    main()
