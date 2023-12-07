from svidreader.video_supplier import VideoSupplier
from enum import Enum
import numpy as np

class Mode(Enum):
    BLINKING = 0
    CONTINUOUS = 1


class LightDetector(VideoSupplier):
    def __init__(self, reader, mode):
        super().__init__(n_frames=reader.n_frames-1, inputs=(reader,))
        self.cache = {}
        if mode == "blinking":
            self.mode = Mode.BLINKING
        elif mode == "continuous":
            self.mode = Mode.CONTINUOUS
        import cupy as cp
        import cupyx.scipy.ndimage

        self.normalize = cp.fuse(LightDetector.get_normalize(cp))
        self.convolve = LightDetector.get_convolve(cp, cupyx.scipy.ndimage)
        self.convolve_big = LightDetector.get_convolve_big(cp, cupyx.scipy.ndimage)


    @staticmethod
    def get_double_gauss(xp):
        def double_gauss(frame):
            return xp.gaussian_filter(frame, sigma=5, truncate=3.5) - xp.gaussian_filter(frame, sigma=2, truncate=5)
        return double_gauss

    @staticmethod
    def get_convolve(xp,ndimg):
        weights_pos = xp.asarray([1, 2, 1])
        weights_neg= xp.asarray([1, 1, 0, 0,0,0, 0, 1, 1])
        def convolve_impl(res):
            res_pos = ndimg.convolve1d(res, weights_pos, axis=0)
            res_pos = ndimg.convolve1d(res_pos, weights_pos, axis=1)
            res_neg = ndimg.convolve1d(res, weights_neg, axis=0)
            res_neg = ndimg.convolve1d(res_neg, weights_neg, axis=1)
            return res_pos - res_neg
        return convolve_impl

    @staticmethod
    def get_convolve_big(xp,ndimg):
        weights_pos = xp.asarray([2, 5, 2])
        weights_neg= xp.asarray([1, 2, 2, 2, 1, 0,0,0,0,0, 1, 2, 2, 2, 1])
        def convolve_big_impl(res):
            res_pos = ndimg.convolve1d(res, weights_pos, axis=0)
            res_pos = ndimg.convolve1d(res_pos, weights_pos, axis=1)
            res_neg = ndimg.convolve1d(res, weights_neg, axis=0)
            res_neg = ndimg.convolve1d(res_neg, weights_neg, axis=1)
            return res_pos - res_neg
        return convolve_big_impl

    @staticmethod
    def get_normalize(xp):
        def normalize_impl(data, divide):
            data = data * (1 / divide)
            data = xp.maximum(data, 0)
            data = xp.sqrt(data)
            return data.astype(xp.uint8)
        return normalize_impl

    def read(self, index, force_type=np):
        import cupy as cp
        xp = cp
        if index in self.cache:
            lastframe = self.cache[index]
        else:
            lastframe = self.inputs[0].read(index=index, force_type=xp)
            lastframe= lastframe.astype(xp.float32)
            lastframe = xp.square(lastframe)

        if self.mode == Mode.BLINKING:
            if index + 1 in self.cache:
                curframe = self.cache[index + 1]
            else:
                curframe = self.inputs[0].read(index=index + 1, force_type=xp)
                curframe = curframe.astype(xp.float32)
                curframe = xp.square(curframe)
        self.cache.clear()
        divide = 81
        self.cache[index] = lastframe
        if self.mode == Mode.BLINKING:
            self.cache[index + 1] = curframe
            res= self.convolve(xp.sum(curframe- lastframe,axis=2))
            res = xp.abs(res)
            divide *= 16 * 3
        else:
            res = lastframe
        res = self.convolve_big(res)
        return VideoSupplier.convert(self.normalize(res, divide), force_type)