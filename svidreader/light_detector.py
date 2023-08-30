from svidreader.video_supplier import VideoSupplier
import numpy as np
from scipy.ndimage import convolve1d
import skimage
from cupyx.scipy.ndimage import gaussian_filter1d
from cupyx.scipy.ndimage import gaussian_filter
import cupy as cp
from cupyx.scipy.ndimage import convolve1d
import cv2

class LightDetector(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))

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
        input = self.inputs[0].read(index=index)
        lastframe = cp.asarray(input)
        curframe = cp.asarray(self.inputs[0].read(index=index + 1))

        lastframe= lastframe.astype(cp.float32)
        curframe = curframe.astype(cp.float32)
        lastframe = cp.square(lastframe)
        curframe = cp.square(curframe)
        res= self.convolve(cp.sum(curframe- lastframe,axis=2) / 3)
        res = cp.abs(res)
        res = self.convolve_big(res)
        res *= 1/(144 * 16)
        res = cp.maximum(res, 0)
        res = cp.sqrt(res)
        return cp.asnumpy(res.astype(cp.uint8))