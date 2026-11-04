import os
import pickle
import tarfile
import urllib.request

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _download(root):
    path = os.path.join(root, "cifar-10-batches-py")
    if os.path.isdir(path):
        return path
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, "cifar-10-python.tar.gz")
    if not os.path.exists(archive):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as t:
        t.extractall(root, filter="data")
    return path


def _batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32), np.array(d[b"labels"], dtype=np.int32)


def load(root="data"):
    path = _download(root)
    train = [_batch(os.path.join(path, f"data_batch_{i}")) for i in range(1, 6)]
    xtr = np.concatenate([b[0] for b in train])
    ytr = np.concatenate([b[1] for b in train])
    xte, yte = _batch(os.path.join(path, "test_batch"))
    return xtr, ytr, xte, yte
