# minnet

A convolutional neural network library written from scratch in C, with Python bindings.
Conv, dense, ReLU, max-pool, softmax cross-entropy, backprop and Adam. No ML frameworks.

Reaches **85.7%** test accuracy on CIFAR-10 (20 epochs, ~26 min on an Apple M3 CPU).

## Build

```sh
make
pip install -r requirements.txt
```

## Train on CIFAR-10

```sh
python python/train.py
```

## Predict

```sh
python python/predict.py models/cifar10.mnet photo.jpg
```

## Test

```sh
python python/gradcheck.py
```
