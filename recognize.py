
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


VALLEY = 0.5


def split_lines(image, margin=4):
    """Split a page image into text-line crops by cutting at ink-density valleys."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = gray.shape

    low, high = np.percentile(gray, 1), np.percentile(gray, 99)
    if high - low < 30:
        return [image]
    ink = gray < (low + high) / 2

    # Ignore the outer edge so page borders and scanner shadows don't count as ink
    edge_y, edge_x = int(height * 0.03), int(width * 0.03)
    inner = np.zeros_like(ink)
    inner[edge_y:height - edge_y, edge_x:width - edge_x] = ink[
        edge_y:height - edge_y, edge_x:width - edge_x
    ]

    profile = inner.sum(axis=1).astype(np.float32)
    if profile.max() == 0:
        return [image]

    # Estimate the line pitch from the profile's autocorrelation
    centered = profile - profile.mean()
    auto = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    lo, hi = 10, min(len(auto) - 1, height // 3)
    if hi <= lo:
        return [image]
    pitch = lo + int(np.argmax(auto[lo:hi]))

    window = max(3, pitch // 4)
    smooth = np.convolve(profile, np.ones(window) / window, mode="same")
    text_level = np.percentile(smooth[smooth > 0], 75)

    # Cuts are ink-density valleys, accepted lowest-first at least ~a line apart
    radius = max(pitch // 3, 1)
    candidates = [
        y for y in range(height)
        if smooth[y] < VALLEY * text_level
        and smooth[y] <= smooth[max(y - radius, 0):y + radius + 1].min()
    ]
    cuts = []
    for y in sorted(candidates, key=lambda i: smooth[i]):
        if all(abs(y - c) >= 0.6 * pitch for c in cuts):
            cuts.append(y)
    cuts.sort()

    bounds = [0] + cuts + [height]
    crops = []
    for top, bottom in zip(bounds[:-1], bounds[1:]):
        band = inner[top:bottom]
        rows = np.where(band.sum(axis=1) > 0)[0]
        cols = np.where(band.any(axis=0))[0]
        if len(rows) < 8 or len(cols) == 0:
            continue
        crops.append(image.crop((
            max(cols[0] - margin, 0),
            max(top + rows[0] - margin, 0),
            min(cols[-1] + margin, width),
            min(top + rows[-1] + margin, height),
        )))
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
