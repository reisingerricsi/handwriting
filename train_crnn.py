"""Train a CNN + BiLSTM + CTC handwriting line recognizer from scratch on IAM."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from iam_dataset import cer, load_iam_lines, split_samples, wer

IMG_HEIGHT = 64
MAX_WIDTH = 1024
CHARSET = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 = CTC blank
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}


def encode(text):
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode_greedy(indices):
    """Collapse repeats and drop blanks."""
    out, prev = [], 0
    for i in indices:
        if i != prev and i != 0:
            out.append(IDX_TO_CHAR[i])
        prev = i
    return "".join(out)


def load_image(path):
    image = Image.open(path).convert("L")
    w, h = image.size
    new_w = min(max(int(w * IMG_HEIGHT / h), 16), MAX_WIDTH)
    image = image.resize((new_w, IMG_HEIGHT), Image.BILINEAR)
    image = ImageOps.invert(image)  # ink = high values, background = 0
    tensor = torch.from_numpy(np.asarray(image, dtype="float32") / 255.0)
    return tensor.unsqueeze(0)  # (1, H, W)


class LineDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        image = load_image(path)
        if self.augment:
            image = self._augment(image)
        return image, torch.tensor(encode(text), dtype=torch.long), text

    @staticmethod
    def _augment(image):
        # Random contrast and small noise
        image = image * (0.8 + 0.4 * torch.rand(1))
        image = image + 0.02 * torch.randn_like(image)
        return image.clamp(0, 1)


def collate(batch):
    images, targets, texts = zip(*batch)
    widths = torch.tensor([im.shape[-1] for im in images])
    # pad along width: permute to (W, 1, H) for pad_sequence, then back
    padded = pad_sequence([im.permute(2, 0, 1) for im in images], batch_first=True)
    padded = padded.permute(0, 2, 3, 1)  # (B, 1, H, W)
    target_lengths = torch.tensor([len(t) for t in targets])
    return padded, widths, torch.cat(targets), target_lengths, list(texts)


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def block(cin, cout, pool):
            layers = [
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(pool))
            return layers

        self.cnn = nn.Sequential(
            *block(1, 32, (2, 2)),    # H 32, W/2
            *block(32, 64, (2, 2)),   # H 16, W/4
            *block(64, 128, None),
            *block(128, 128, (2, 1)),  # H 8, W/4
            *block(128, 256, None),
            *block(256, 256, (2, 1)),  # H 4, W/4
            nn.Dropout2d(0.2),
        )
        self.rnn = nn.LSTM(
            256 * (IMG_HEIGHT // 16), 256, num_layers=2,
            bidirectional=True, batch_first=True, dropout=0.25,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feats = self.cnn(x)  # (B, C, H', W')
        b, c, h, w = feats.shape
        feats = feats.permute(0, 3, 1, 2).reshape(b, w, c * h)
        out, _ = self.rnn(feats)
        return self.fc(out)  # (B, W', classes)

    @staticmethod
    def output_lengths(widths):
        return widths // 4


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, refs = [], []
    for images, widths, _, _, texts in loader:
        logits = model(images.to(device))
        best = logits.argmax(-1).cpu()
        lengths = CRNN.output_lengths(widths)
        for seq, n, ref in zip(best, lengths, texts):
            preds.append(decode_greedy(seq[:n].tolist()))
            refs.append(ref)
    return cer(preds, refs), wer(preds, refs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iam-root", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", default="crnn.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = split_samples(load_iam_lines(args.iam_root))
    print(f"train={len(train)} val={len(val)} test={len(test)} device={device}")

    def loader(samples, shuffle, augment=False):
        return DataLoader(
            LineDataset(samples, augment), batch_size=args.batch_size,
            shuffle=shuffle, collate_fn=collate, num_workers=args.workers,
        )

    train_loader = loader(train, True, augment=True)
    val_loader = loader(val, False)
    test_loader = loader(test, False)

    model = CRNN(len(CHARSET) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader)
    )
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best_cer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, widths, targets, target_lengths, _ in train_loader:
            logits = model(images.to(device))
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)  # (T, B, C)
            loss = ctc(
                log_probs, targets, CRNN.output_lengths(widths), target_lengths
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()

        val_cer, val_wer = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: loss={total / len(train_loader):.4f} "
              f"val CER={val_cer:.4f} WER={val_wer:.4f}")
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), args.output)

    model.load_state_dict(torch.load(args.output, map_location=device))
    test_cer, test_wer = evaluate(model, test_loader, device)
    print(f"test CER={test_cer:.4f} WER={test_wer:.4f} (saved to {Path(args.output)})")


if __name__ == "__main__":
    main()
