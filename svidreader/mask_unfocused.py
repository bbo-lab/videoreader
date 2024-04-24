from svidreader.video_supplier import VideoSupplier
import numpy as np
import scipy.stats as stats

class NormalizedContrast(VideoSupplier):
    def __init__(self, reader, options = {}):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.lib = options.get('lib', "nb")
        if self.lib == 'cupy':
            import cupy as cp
            import cupyx.scipy.ndimage
            self.sqnorm = cp.fuse(NormalizedContrast.sqnorm(cp))
            ndimage = cupyx.scipy.ndimage
            self.xp = cp
        elif self.lib == 'jax':
            import jax
            self.sqnorm = jax.jit(NormalizedContrast.sqnorm(jax.numpy))
            self.xp = jax.numpy
        elif self.lib == 'nb':
            import numba as nb
            self.sqnorm = nb.jit(NormalizedContrast.sqnorm(np))
            self.xp = np
        else:
            self.sqnorm = sqnorm(np)
            self.xp = np
        self.convolve = NormalizedContrast.get_convolve(self.xp, ndimage, width=20, normalize=1/2)
        #self.convolve = NormalizedContrast.get_convolve(self.xp, ndimage)


    @staticmethod
    def get_convolve(xp,ndimg, width=50, normalize=np.nan):
        weights= xp.asarray(stats.norm.pdf(np.linspace(-2,2,width), 0, 1),dtype=xp.float32)
        if not np.isnan(normalize):
            weights *= normalize / xp.sum(weights)
        def convolve_impl(res):
            res = ndimg.convolve1d(res, weights, axis=0)
            res = ndimg.convolve1d(res, weights, axis=1)
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
        img = self.xp.square(img)
        gy, gx = self.xp.gradient(img, axis=(0, 1))
        contrast = self.sqnorm(gx, gy)
        gy, gx = self.xp.gradient(self.convolve(img), axis=(0, 1))
        contrast = self.convolve(contrast) / (self.sqnorm(gx, gy) + 5)
        contrast *= 10
        contrast = self.xp.minimum(contrast, 255)
        contrast = contrast.astype(self.xp.uint8)
        return VideoSupplier.convert(contrast, force_type)

    def read_(self, index, force_type=np):
        img = self.inputs[0].read(index=index, force_type=self.xp)
        img = self.xp.asarray(img, dtype=self.xp.float32)
        img_blurred = self.convolve(img)
        gy, gx = self.xp.gradient(img, axis=(0, 1))
        contrast = self.sqnorm(gx, gy)
        gy, gx = self.xp.gradient(img_blurred, axis=(0, 1))
        contrast /= (self.sqnorm(gx, gy) + 1)
        return VideoSupplier.convert(contrast, force_type)