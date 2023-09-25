from numba import jit
from svidreader.video_supplier import VideoSupplier
import numpy as np
from functools import partial

@jit(nopython=True, fastmath=True)
def gradient2d(f=[[[]]]):
    out = np.empty_like(f, np.float32)
    out[1:-1] = (f[2:] - f[:-2])
    out[0] = (f[1] - f[0]) * 2
    out[-1] = (f[-1] - f[-2]) * 2
    return out


@jit(nopython=True, fastmath=True)
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
        else:
            self.sqnorm = sqnorm(np)


    def read(self, index):
        img = self.inputs[0].read(index=index)
        if self.lib == 'cupy':
            import cupy as cp
            img = cp.asarray(img, dtype=cp.float32)
            gy, gx = cp.gradient(img, axis=(0, 1))
            return {'contrast':cp.average(self.sqnorm(gx,gy)), 'brightness':cp.average(img)}
        if self.lib == 'jax':
            import jax
            img = jax.numpy.asarray(img, dtype = jax.numpy.float32)
            img = jax.device_put(img, device=jax.devices('gpu')[0])
            gy, gx = jax.numpy.gradient(img, axis=(0, 1))
            return {'contrast':jax.numpy.average(self.sqnorm(gx,gy)), 'brightness':jax.numpy.average(img)}

        if False:
            cr, av = analyze(img.astype(np.float32))
            return {'contrast': cr, 'brightness': av}
        gy, gx = np.gradient(img.astype(np.float32), axis=(0, 1))
        gx = self.sqnorm(gx, gy)
        return {'contrast':np.average(gx), 'brightness':np.average(img)}