from numba import jit
from svidreader.video_supplier import VideoSupplier
import numpy as np


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


class AnalyzeImage(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))


    def read(self, index):
        img = self.inputs[0].read(index=index)
        if False:
            cr, av = analyze(img.astype(np.float32))
            return {'contrast': cr, 'brightness': av}
        gy, gx = np.gradient(img.astype(np.float32), axis=(0, 1))
        np.square(gx, out=gx)
        np.square(gy, out=gy)
        gx += gy
        np.sqrt(gx, out=gx)
        return {'contrast':np.average(gx), 'brightness':np.average(img)}