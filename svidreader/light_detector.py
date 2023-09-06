from svidreader.video_supplier import VideoSupplier
import numpy as np
from scipy.ndimage import convolve1d
import skimage
from cupyx.scipy.ndimage import gaussian_filter1d
from cupyx.scipy.ndimage import gaussian_filter
import cupy as cp
from cupyx.scipy.ndimage import convolve1d
import cv2
from enum import Enum


class Mode(Enum):
    BLINKING = 0
    CONTINUOUS = 1


class LightDetector(VideoSupplier):
    def __init__(self, reader, mode):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.cache = {}
        if mode == "blinking":
            self.mode = Mode.BLINKING
        elif mode == "continuous":
            self.mode = Mode.CONTINUOUS

    @staticmethod
    def double_gauss(frame):
        return gaussian_filter(frame, sigma=5, truncate=3.5) - gaussian_filter(frame, sigma=2, truncate=5)


    @staticmethod
    def convolve(res):
        weights_pos = cp.asarray([1, 2, 1])
        weights_neg= cp.asarray([1, 1, 0, 0,0,0, 0, 1, 1])
        res_pos = convolve1d(res, weights_pos, axis=0)
        res_pos = convolve1d(res_pos, weights_pos, axis=1)
        res_neg = convolve1d(res, weights_neg, axis=0)
        res_neg = convolve1d(res_neg, weights_neg, axis=1)
        return res_pos - res_neg

    @staticmethod
    def convolve_big(res):
        weights_pos = cp.asarray([2, 5, 2])
        weights_neg= cp.asarray([1, 2, 2, 2, 1, 0,0,0,0,0, 1, 2, 2, 2, 1])
        res_pos = convolve1d(res, weights_pos, axis=0)
        res_pos = convolve1d(res_pos, weights_pos, axis=1)
        res_neg = convolve1d(res, weights_neg, axis=0)
        res_neg = convolve1d(res_neg, weights_neg, axis=1)
        return res_pos - res_neg


    def read(self, index):
        if index in self.cache:
            lastframe = self.cache[index]
        else:
            lastframe = cp.asarray(self.inputs[0].read(index=index))
            lastframe= lastframe.astype(cp.float32)
            lastframe = cp.square(lastframe)

        if self.mode == Mode.BLINKING:
            if index + 1 in self.cache:
                curframe = self.cache[index + 1]
            else:
                curframe = cp.asarray(self.inputs[0].read(index=index + 1))
                curframe = curframe.astype(cp.float32)
                curframe = cp.square(curframe)
        self.cache.clear()
        divide = 81
        self.cache[index] = lastframe
        if self.mode == Mode.BLINKING:
            self.cache[index + 1] = curframe
            res= self.convolve(cp.sum(curframe- lastframe,axis=2) / 3)
            res = cp.abs(res)
            divide *= 16
        else:
            res = lastframe
        res = self.convolve_big(res)
        res *= 1/divide
        res = cp.maximum(res, 0)
        res = cp.sqrt(res)
        return cp.asnumpy(res.astype(cp.uint8))