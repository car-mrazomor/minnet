import ctypes
import os
import sys

import numpy as np

_F = ctypes.POINTER(ctypes.c_float)
_I = ctypes.POINTER(ctypes.c_int)
_P = ctypes.c_void_p


def _load():
    name = "libminnet.dylib" if sys.platform == "darwin" else "libminnet.so"
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.join(here, "..", "build")):
        path = os.path.join(d, name)
        if os.path.exists(path):
            return ctypes.CDLL(path)
    raise OSError(f"{name} not found, run make first")


lib = _load()

for fn, res, args in [
    ("mn_new", _P, [ctypes.c_int] * 3),
    ("mn_free", None, [_P]),
    ("mn_conv", ctypes.c_int, [_P] + [ctypes.c_int] * 4),
    ("mn_dense", ctypes.c_int, [_P, ctypes.c_int]),
    ("mn_relu", ctypes.c_int, [_P]),
    ("mn_maxpool", ctypes.c_int, [_P, ctypes.c_int, ctypes.c_int]),
    ("mn_init", None, [_P, ctypes.c_uint]),
    ("mn_set_norm", None, [_P, _F, _F]),
    ("mn_input_shape", None, [_P, _I, _I, _I]),
    ("mn_input_size", ctypes.c_int, [_P]),
    ("mn_output_size", ctypes.c_int, [_P]),
    ("mn_layer_count", ctypes.c_int, [_P]),
    ("mn_param_count", ctypes.c_int, [_P, ctypes.c_int]),
    ("mn_params", _F, [_P, ctypes.c_int]),
    ("mn_grads", _F, [_P, ctypes.c_int]),
    ("mn_forward", _F, [_P, _F, ctypes.c_int]),
    ("mn_backward", ctypes.c_float, [_P, _I, ctypes.c_int]),
    ("mn_step", None, [_P, ctypes.c_float, ctypes.c_float]),
    ("mn_train", ctypes.c_float, [_P, _F, _I, ctypes.c_int, ctypes.c_float, ctypes.c_float]),
    ("mn_save", ctypes.c_int, [_P, ctypes.c_char_p]),
    ("mn_load", _P, [ctypes.c_char_p]),
]:
    f = getattr(lib, fn)
    f.restype = res
    f.argtypes = args


def _f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def _i32(y):
    return np.ascontiguousarray(y, dtype=np.int32)


class Net:
    def __init__(self, c, h, w, _ptr=None):
        self.ptr = _ptr if _ptr else lib.mn_new(c, h, w)
        if not self.ptr:
            raise ValueError("invalid input shape")

    @classmethod
    def load(cls, path):
        ptr = lib.mn_load(os.fsencode(path))
        if not ptr:
            raise OSError(f"cannot load {path}")
        return cls(0, 0, 0, _ptr=ptr)

    def __del__(self):
        if getattr(self, "ptr", None):
            lib.mn_free(self.ptr)
            self.ptr = None

    def _add(self, r, what):
        if r:
            raise ValueError(f"cannot add {what} layer")
        return self

    def conv(self, filters, size=3, stride=1, pad=None):
        pad = size // 2 if pad is None else pad
        return self._add(lib.mn_conv(self.ptr, filters, size, stride, pad), "conv")

    def dense(self, outputs):
        return self._add(lib.mn_dense(self.ptr, outputs), "dense")

    def relu(self):
        return self._add(lib.mn_relu(self.ptr), "relu")

    def maxpool(self, size=2, stride=None):
        return self._add(lib.mn_maxpool(self.ptr, size, stride or size), "maxpool")

    def init(self, seed=0):
        lib.mn_init(self.ptr, seed)
        return self

    def set_norm(self, mean, std):
        c = self.input_shape[0]
        mean, std = _f32(mean), _f32(std)
        if mean.size != c or std.size != c:
            raise ValueError(f"expected {c} values")
        lib.mn_set_norm(self.ptr, mean.ctypes.data_as(_F), std.ctypes.data_as(_F))

    @property
    def input_shape(self):
        c, h, w = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        lib.mn_input_shape(self.ptr, ctypes.byref(c), ctypes.byref(h), ctypes.byref(w))
        return c.value, h.value, w.value

    @property
    def output_size(self):
        return lib.mn_output_size(self.ptr)

    @property
    def layers(self):
        return lib.mn_layer_count(self.ptr)

    def _view(self, fn, layer):
        n = lib.mn_param_count(self.ptr, layer)
        return np.ctypeslib.as_array(fn(self.ptr, layer), shape=(n,)) if n else None

    def params(self, layer):
        return self._view(lib.mn_params, layer)

    def grads(self, layer):
        return self._view(lib.mn_grads, layer)

    def _batch(self, x):
        x = _f32(x)
        n = x.shape[0]
        if x.size != n * lib.mn_input_size(self.ptr):
            raise ValueError(f"expected input shape (n, {self.input_shape})")
        return x, n

    def forward(self, x):
        x, n = self._batch(x)
        p = lib.mn_forward(self.ptr, x.ctypes.data_as(_F), n)
        if not p:
            raise MemoryError
        return np.ctypeslib.as_array(p, shape=(n, self.output_size)).copy()

    def backward(self, y):
        y = _i32(y)
        return lib.mn_backward(self.ptr, y.ctypes.data_as(_I), len(y))

    def step(self, lr, wd=0.0):
        lib.mn_step(self.ptr, lr, wd)

    def train(self, x, y, lr, wd=0.0):
        x, n = self._batch(x)
        y = _i32(y)
        if len(y) != n:
            raise ValueError("x and y sizes differ")
        return lib.mn_train(self.ptr, x.ctypes.data_as(_F), y.ctypes.data_as(_I), n, lr, wd)

    def predict(self, x, batch=64):
        return np.concatenate([self.forward(x[i:i + batch]).argmax(1) for i in range(0, len(x), batch)])

    def save(self, path):
        if lib.mn_save(self.ptr, os.fsencode(path)):
            raise OSError(f"cannot save {path}")


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
