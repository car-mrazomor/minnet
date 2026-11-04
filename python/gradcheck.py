import sys

import numpy as np

from minnet import Net


def main():
    rng = np.random.default_rng(1)
    net = Net(2, 9, 9)
    net.conv(4, 3).relu().conv(3, 3, stride=2, pad=1).maxpool(2).dense(6).relu().dense(5)
    net.init(3)
    x = rng.normal(size=(4, 2, 9, 9)).astype(np.float32)
    y = rng.integers(0, 5, 4)

    net.forward(x)
    net.backward(y)
    eps = 1e-3
    worst = 0.0
    for layer in range(net.layers):
        p = net.params(layer)
        if p is None:
            continue
        g = net.grads(layer).copy()
        num = np.zeros_like(g)
        for i in range(p.size):
            old = p[i]
            p[i] = old + eps
            net.forward(x)
            up = net.backward(y)
            p[i] = old - eps
            net.forward(x)
            down = net.backward(y)
            p[i] = old
            num[i] = (up - down) / (2 * eps)
        err = np.linalg.norm(num - g) / (np.linalg.norm(num) + np.linalg.norm(g))
        worst = max(worst, err)
        print(f"layer {layer}: {p.size} params, relative error {err:.2e}")
    print(f"max relative error {worst:.2e}")
    sys.exit(0 if worst < 1e-2 else 1)


if __name__ == "__main__":
    main()
