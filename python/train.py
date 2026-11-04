import argparse
import math
import os
import time

import numpy as np

import cifar
from minnet import Net


def build():
    net = Net(3, 32, 32)
    for f in (32, 64, 128):
        net.conv(f).relu().conv(f).relu().maxpool()
    return net.dense(256).relu().dense(10)


def scale(x):
    return x.astype(np.float32) / 255.0


def augment(x, rng):
    n = len(x)
    flip = rng.random(n) < 0.5
    x = np.where(flip[:, None, None, None], x[..., ::-1], x)
    p = np.pad(x, ((0, 0), (0, 0), (4, 4), (4, 4)), mode="reflect")
    ox, oy = rng.integers(0, 9, n), rng.integers(0, 9, n)
    return np.stack([p[i, :, oy[i]:oy[i] + 32, ox[i]:ox[i] + 32] for i in range(n)])


def evaluate(net, x, y, batch):
    hits = 0
    for i in range(0, len(x), batch):
        hits += int((net.forward(scale(x[i:i + batch])).argmax(1) == y[i:i + batch]).sum())
    return hits / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/cifar10.mnet")
    args = ap.parse_args()

    xtr, ytr, xte, yte = cifar.load(args.data)
    net = build().init(args.seed)
    mean = [scale(xtr[:, c]).mean() for c in range(3)]
    std = [scale(xtr[:, c]).std() for c in range(3)]
    net.set_norm(mean, std)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(os.path.splitext(args.out)[0] + ".labels", "w") as f:
        f.write("\n".join(cifar.CLASSES) + "\n")

    rng = np.random.default_rng(args.seed)
    per_epoch = len(xtr) // args.batch
    total = args.epochs * per_epoch
    step = 0
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        order = rng.permutation(len(xtr))
        loss = 0.0
        for i in range(per_epoch):
            idx = order[i * args.batch:(i + 1) * args.batch]
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / total))
            loss += net.train(augment(scale(xtr[idx]), rng), ytr[idx], lr, args.wd)
            step += 1
        acc = evaluate(net, xte, yte, args.batch)
        net.save(args.out)
        print(f"epoch {epoch:3d}  loss {loss / per_epoch:.4f}  test {acc:.4f}  {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
