from svidreader.video_supplier import VideoSupplier
import numpy as np
import scipy.stats as stats

class RadialContrast(VideoSupplier):
    def __init__(
            self,
            reader:VideoSupplier,
            options=None,
            width=20,
            kernel=("normalized", "gaussian"),
            normalize=50):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        if options == None:
            options = {}
        self.full_vector = options.get('full_vector', False)
        self.lib = options.get('lib', "cupy")
        if self.lib == 'cupy':
            import cupy as cp
            import cupyx.scipy.ndimage
            sqnorm = cp.fuse(RadialContrast.sqnorm(cp))
            conv = cupyx.scipy.ndimage.convolve
            self.xp = cp
        elif self.lib == 'jax':
            import jax
            sqnorm = jax.jit(RadialContrast.sqnorm(jax.numpy))
            conv = jax.scipy.signal.convolve
            self.xp = jax.numpy
        elif self.lib == 'nb':
            import numba as nb
            import scipy
            sqnorm = nb.jit(RadialContrast.sqnorm(np))
            conv = scipy.ndimage.convolve
            self.xp = np
        elif self.lib == None or self.lib == 'numpy' or self.lib == 'np':
            import scipy
            sqnorm = RadialContrast.sqnorm(np)
            conv = scipy.ndimage.convolve
            self.xp = np
        else:
            raise ValueError(f"Unknown library {self.lib}")
        self.convolve = RadialContrast.get_convolve(
            self.xp,
            conv,
            sqnorm,
            width=width,
            kernel=kernel,
            normalize=normalize,
            full_vector=self.full_vector)

    @staticmethod
    def get_convolve(xp,
                     conv,
                     sqnorm,
                     width=20,
                     kernel=("gaussian",),
                     normalize=np.nan,
                     full_vector=False):
        xx, yy = np.mgrid[-1:1:width * 1j, -1:1:width * 1j]
        atan = np.arctan2(yy, xx) * 2
        w0 = np.sin(atan)
        w1 = np.cos(atan)
        d = np.sqrt(np.square(xx) + np.square(yy))
        mask = d <= 1.01
        #0.31
        wpeak = np.copy(mask).astype(np.float32)
        wpeak *= np.asarray((stats.norm.pdf(d, 0, 1) - stats.norm.pdf(1, 0, 1)), dtype=np.float32)
        mask = mask.astype(np.float32)
        if "gaussian" in kernel:
            mask *= np.asarray((stats.norm.pdf(d, 0, 1) - stats.norm.pdf(1, 0, 1)), dtype=np.float32)
        elif "normalized" in kernel:
            #Every distance adds the same weight, so we have do divide by the distance
            mask *= np.asarray((1 / (d + 1e-6)), dtype=np.float32)
        else:
            raise ValueError(f"Unknown kernel {kernel}")
        w0 *= mask
        w1 *= mask
        w0 = xp.asarray(w0, dtype=xp.float32)
        w1 = xp.asarray(w1, dtype=xp.float32)
        wpeak = xp.asarray(wpeak, dtype=xp.float32)
        if not np.isnan(normalize):
            w0 *= 1 / xp.sum(xp.abs(w0))
            w1 *= 1 / xp.sum(xp.abs(w1))
            wpeak *= normalize / xp.sum(xp.abs(wpeak))

        def convolve_impl(res):
            res = xp.sum(res, axis=2)
            res = conv(res, w0), conv(res, w1)
            if full_vector:
                return xp.stack(res, axis=-1)
            res = sqnorm(*res)
            #res = xp.maximum(conv(res, wpeak), 0)
            return res

        plot_stamp = False
        if plot_stamp:
            from matplotlib import pyplot as plt
            stacked = xp.abs(xp.stack([w0, w1, wpeak], axis=-1))
            stacked /= np.max(stacked, axis=(0, 1), keepdims=True)
            plt.imshow(VideoSupplier.convert(stacked, np))
            plt.show()

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
        if not self.full_vector:
        #img = self.xp.minimum(img, 255)
            img = img * (255 / self.xp.max(img))
            img = self.xp.asarray(img, dtype=self.xp.uint8)
        return VideoSupplier.convert(img, force_type)
