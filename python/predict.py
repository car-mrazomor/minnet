import argparse
import os

import numpy as np
from PIL import Image

from minnet import Net, softmax


def prepare(path, w, h):
    img = Image.open(path).convert("RGB")
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    net = Net.load(args.model)
    _, h, w = net.input_shape
    labels_path = os.path.splitext(args.model)[0] + ".labels"
    with open(labels_path) as f:
        labels = f.read().split()

    for path in args.images:
        p = softmax(net.forward(prepare(path, w, h))[0])
        print(path)
        for i in p.argsort()[::-1][:args.top]:
            print(f"  {labels[i]:<12} {p[i]:.3f}")


if __name__ == "__main__":
    main()
