from decord import VideoReader
import numpy as np
from svidreader.video_supplier import VideoSupplier

class DecordVideoReader(VideoSupplier):
    def __init__(self, filename):
        self.vr = VideoReader(filename)
        super().__init__(n_frames=len(self.vr), inputs=())


    def read(self, index):
        res = self.vr.get_batch([index]).asnumpy()
        res = res[0]
        return res