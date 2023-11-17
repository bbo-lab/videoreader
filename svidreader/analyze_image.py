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
        elif self.lib == 'jax':
            import jax
            self.sqnorm = jax.jit(sqnorm(jax.numpy))
        elif self.lib == 'nb':
            import numba as nb
            self.sqnorm = nb.jit(sqnorm(np))
        else:
            self.sqnorm = sqnorm(np)


    def read(self, index):
        img = self.inputs[0].read(index=index)
        contrast = 0
        brightness = 0
        if self.lib == 'cupy':
            import cupy as cp
            img = cp.asarray(img, dtype=cp.float32)
            gy, gx = cp.gradient(img, axis=(0, 1))
            contrast = cp.average(self.sqnorm(gx,gy))
            brightness = cp.average(img)
        elif self.lib == 'jax':
            import jax
            img = jax.numpy.asarray(img, dtype = jax.numpy.float32)
            img = jax.device_put(img, device=jax.devices('gpu')[0])
            gy, gx = jax.numpy.gradient(img, axis=(0, 1))
            contrast = jax.numpy.average(self.sqnorm(gx,gy))
            brightness = jax.numpy.average(img)
        elif self.lib == 'nb':
            import numba as nb
            self.analyze = nb.jit(analyze)
            cr, av = analyze(img.astype(np.float32))
            contrast = cr
            brightness = av
        else:
            gy, gx = np.gradient(img.astype(np.float32), axis=(0, 1))
            gx = self.sqnorm(gx, gy)
            contrast = np.average(gx)
            brightness = np.average(img)
        return {'contrast': contrast, 'brightness': brightness}
