"""IAM Handwriting Database line-level loading and splitting.

Expected layout (download from https://fki.tic.unibe.ch/databases/iam-handwriting-database,
free registration required):

    <iam_root>/
        ascii/lines.txt
        lines/a01/a01-000u/a01-000u-00.png   (line images; .png)
"""

import random
from pathlib import Path


def load_iam_lines(iam_root):
    """Return a list of (image_path, transcription) for all usable lines."""
    iam_root = Path(iam_root)
    lines_txt = iam_root / "ascii" / "lines.txt"
    if not lines_txt.is_file():
        raise FileNotFoundError(f"Missing {lines_txt}")

    samples = []
    for raw in lines_txt.read_text(encoding="utf-8").splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        parts = raw.split(" ")
        line_id, status = parts[0], parts[1]
        if status != "ok":
            continue
        # IAM uses '|' as the word separator in transcriptions
        text = " ".join(parts[8:]).replace("|", " ").strip()
        form = line_id.rsplit("-", 1)[0]
        writer_dir = line_id.split("-")[0]
        image_path = iam_root / "lines" / writer_dir / form / f"{line_id}.png"
        if image_path.is_file() and text:
            samples.append((image_path, text))
    if not samples:
        raise RuntimeError(f"No samples found under {iam_root}")
    return samples


def split_samples(samples, val_frac=0.1, test_frac=0.1, seed=42):
    """Split by form so lines from one page never leak across splits."""
    forms = sorted({p.parent.name for p, _ in samples})
    random.Random(seed).shuffle(forms)
    n_val = int(len(forms) * val_frac)
    n_test = int(len(forms) * test_frac)
    val_forms = set(forms[:n_val])
    test_forms = set(forms[n_val:n_val + n_test])

    train, val, test = [], [], []
    for sample in samples:
        form = sample[0].parent.name
        if form in val_forms:
            val.append(sample)
        elif form in test_forms:
            test.append(sample)
        else:
            train.append(sample)
    return train, val, test


def edit_distance(a, b):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(predictions, references):
    """Character error rate over a list of strings."""
    errors = sum(edit_distance(p, r) for p, r in zip(predictions, references))
    total = sum(len(r) for r in references)
    return errors / max(total, 1)


def wer(predictions, references):
    """Word error rate over a list of strings."""
    errors = sum(edit_distance(p.split(), r.split()) for p, r in zip(predictions, references))
    total = sum(len(r.split()) for r in references)
    return errors / max(total, 1)
