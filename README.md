# Handwriting recognition

Line-level handwriting recognition with neural networks, trained on the
[IAM Handwriting Database](https://fki.tic.unibe.ch/databases/iam-handwriting-database)
(free registration required).

## Setup

```bash
pip install -r requirements.txt
```

Download IAM `lines.tgz` and `ascii.tgz` and extract so you have:

```
<iam_root>/ascii/lines.txt
<iam_root>/lines/a01/a01-000u/a01-000u-00.png
```

Splits are made by form (90/10/10-style random split, seed 42), so no page appears in two splits.

## Scripts

| Script | Purpose |
|---|---|
| `recognize.py` | Pretrained TrOCR inference; `--lines` splits multi-line pages |
| `train_crnn.py` | Train a CNN + BiLSTM + CTC model from scratch |
| `predict_crnn.py` | Run a trained CRNN checkpoint on one image |
| `finetune_trocr.py` | Fine-tune `microsoft/trocr-base-handwritten` on IAM |

```bash
python train_crnn.py --iam-root /path/to/iam --epochs 50
python predict_crnn.py line.png --checkpoint crnn.pt

python finetune_trocr.py --iam-root /path/to/iam --epochs 3
```

Both trainers report validation/test character and word error rates (CER/WER).
A GPU is strongly recommended; TrOCR fine-tuning needs roughly 8+ GB of VRAM at batch size 4.

## Multi-line pages

```bash
python recognize.py page.png --lines
```

Lines are found by horizontal gaps between ink, so use straight, plain-paper pages.

Images are converted to grayscale, lighting is flattened and contrast normalized before recognition. Use `--raw` to skip this.

If faint strokes get lost, add `--gentle`. Use `--save-preprocessed` to write `<name>_preprocessed.png` next to the input and inspect it.
