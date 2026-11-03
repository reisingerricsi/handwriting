"""Fine-tune TrOCR on IAM handwriting lines."""

import argparse

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from iam_dataset import cer, load_iam_lines, split_samples, wer

MODEL_NAME = "microsoft/trocr-base-handwritten"
MAX_LABEL_LEN = 128


class TrOCRDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=MAX_LABEL_LEN, truncation=True
        ).input_ids
        # -100 makes the loss ignore padding
        labels = [t if t != self.processor.tokenizer.pad_token_id else -100 for t in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels), "text": text}


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "text": [b["text"] for b in batch],
    }


@torch.no_grad()
def evaluate(model, processor, loader, device, max_batches=None):
    model.eval()
    preds, refs = [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        ids = model.generate(batch["pixel_values"].to(device), max_new_tokens=MAX_LABEL_LEN)
        preds += processor.batch_decode(ids, skip_special_tokens=True)
        refs += batch["text"]
    return cer(preds, refs), wer(preds, refs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iam-root", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output", default="trocr-iam")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    train, val, test = split_samples(load_iam_lines(args.iam_root))
    print(f"train={len(train)} val={len(val)} test={len(test)} device={device}")

    def loader(samples, shuffle):
        return DataLoader(
            TrOCRDataset(samples, processor), batch_size=args.batch_size,
            shuffle=shuffle, collate_fn=collate, num_workers=2,
        )

    train_loader = loader(train, True)
    val_loader = loader(val, False)
    test_loader = loader(test, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_cer = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader, 1):
            loss = model(
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device),
            ).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={total / step:.4f}")

        val_cer, val_wer = evaluate(model, processor, val_loader, device)
        print(f"epoch {epoch}: val CER={val_cer:.4f} WER={val_wer:.4f}")
        if val_cer < best_cer:
            best_cer = val_cer
            model.save_pretrained(args.output)
            processor.save_pretrained(args.output)

    best = VisionEncoderDecoderModel.from_pretrained(args.output).to(device)
    test_cer, test_wer = evaluate(best, processor, test_loader, device)
    print(f"test CER={test_cer:.4f} WER={test_wer:.4f} (saved to {args.output}/)")


if __name__ == "__main__":
    main()
