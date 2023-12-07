from svidreader.video_supplier import VideoSupplier
import numpy as np
import scipy.stats as stats


class RadialContrast(VideoSupplier):
    def __init__(self, reader, options={}):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.lib = options.get('lib', "cupy")
        if self.lib == 'cupy':
            import cupy as cp
            import cupyx.scipy.ndimage
            self.sqnorm = cp.fuse(RadialContrast.sqnorm(cp))
            ndimage = cupyx.scipy.ndimage
            self.xp = cp
        elif self.lib == 'jax':
            import jax
            self.sqnorm = jax.jit(RadialContrast.sqnorm(jax.numpy))
            self.xp = jax.numpy
        elif self.lib == 'nb':
            import numba as nb
            self.sqnorm = nb.jit(RadialContrast.sqnorm(np))
            self.xp = np
        else:
            import scipy
            self.sqnorm = RadialContrast.sqnorm(np)
            ndimage = scipy.ndimage
            self.xp = np
        self.convolve = RadialContrast.get_convolve(self.xp, ndimage, width=20, normalize=50)

    @staticmethod
    def get_convolve(xp, ndimg, width=20, normalize=np.nan):
        xx, yy = np.mgrid[-1:1:width * 1j, -1:1:width * 1j]
        w0 = np.sin(np.arctan2(yy, xx) * 2)
        w1 = np.cos(np.arctan2(yy, xx) * 2)
        d = np.sqrt(np.square(xx) + np.square(yy))
        mask = d < 1.1
        wpeak = xp.asarray((stats.norm.pdf(d, 0, 1) - 0.31) * mask, dtype=xp.float32)
        mask = mask * stats.norm.pdf(d, 0, 1)
        w0 = w0 * mask
        w1 = w1 * mask
        w0 = xp.asarray(w0, dtype=xp.float32)
        w1 = xp.asarray(w1, dtype=xp.float32)
        if not np.isnan(normalize):
            w0 *= 1 / xp.sum(xp.abs(w0))
            w1 *= 1 / xp.sum(xp.abs(w1))
            wpeak *= normalize / xp.sum(xp.abs(wpeak))

        def convolve_impl(res):
            res = xp.sum(res, axis=2)
            res = xp.sqrt(xp.square(ndimg.convolve(res, w0)) + xp.square(ndimg.convolve(res, w1)))
            res = xp.maximum(ndimg.convolve(res, wpeak), 0)
            return res

        return convolve_impl

    @staticmethod
    def sqnorm(xp):
        def f(gx, gy):
            gx = xp.square(gx)
            gy = xp.square(gy)
            res = gx + gy
            res = xp.sqrt(res)
            return res

        return f

    def read(self, index, force_type=np):
        img = self.inputs[0].read(index=index, force_type=self.xp)
        img = self.xp.asarray(img, dtype=self.xp.float32)
        img = self.convolve(img)
        return VideoSupplier.convert(self.xp.asarray(img, dtype=self.xp.uint8), force_type)
