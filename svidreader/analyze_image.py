from svidreader.video_supplier import VideoSupplier
import numpy as np
from functools import partial

def gradient2d(f=[[[]]]):
    out = np.empty_like(f, np.float32)
    out[1:-1] = (f[2:] - f[:-2])
    out[0] = (f[1] - f[0]) * 2
    out[-1] = (f[-1] - f[-2]) * 2
    return out


def analyze(img=[[[]]]):
    gx = gradient2d(img)
    gy = gradient2d(img.T).T
    gx = np.square(gx)
    gy = np.square(gy)
    gx += gy
    gx = np.sqrt(gx)
    return np.average(gx), np.average(img)


def sqnorm(xp):
    def f(gx, gy):
        gx = xp.square(gx)
        gy = xp.square(gy)
        res = gx + gy
        res = xp.sqrt(res)
        return res
    return f


class AnalyzeImage(VideoSupplier):
    def __init__(self, reader, options = {}):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.lib =options.get('lib',"cupy")
        if self.lib == 'cupy':
            import cupy as cp
            self.sqnorm = cp.fuse(sqnorm(cp))
            self.convolve = AnalyzeImage.get_convolve(cp, cupyx.scipy.ndimage)
            self.xp = cp
        elif self.lib == 'jax':
            import jax
            self.sqnorm = jax.jit(sqnorm(jax.numpy))
            self.xp = jax.numpy
        elif self.lib == 'nb':
            import numba as nb
            self.sqnorm = nb.jit(sqnorm(np))
            self.analyze = nb.jit(analyze)
            self.xp = np
        else:
            self.sqnorm = sqnorm(np)
            self.xp = np


    def read(self, index):
        img = self.inputs[0].read(index=index, force_type=self.xp)
        if self.lib == 'nb':
            import numba as nb
            contrast, brightness = analyze(img.astype(self.xp.float32))
        else:
            img = self.xp.asarray(img, dtype=self.xp.float32)
            gy, gx = self.xp.gradient(img, axis=(0, 1))
            contrast = self.xp.average(self.sqnorm(gx, gy))
            brightness = self.xp.average(img)
        return {'contrast': contrast, 'brightness': brightness}
