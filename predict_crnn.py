"""Run a trained CRNN checkpoint on a handwritten line image."""

import argparse

import torch

from train_crnn import CHARSET, CRNN, decode_greedy, load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="crnn.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(len(CHARSET) + 1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    image = load_image(args.image).unsqueeze(0).to(device)
    with torch.inference_mode():
        best = model(image).argmax(-1)[0].tolist()
    print(decode_greedy(best))


if __name__ == "__main__":
    main()
