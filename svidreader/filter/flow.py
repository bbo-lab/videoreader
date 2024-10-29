import numpy as np
from IPython.core.display import Video

from svidreader.video_supplier import VideoSupplier


class OpticFlow(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.cache = (None,None)


    def read(self, index, force_type=np):
        import cv2 as cv
        if self.cache[0] == index:
            previous = self.cache[1]
        else:
            previous = self.inputs[0].read(index=index, force_type=np)
        current = self.inputs[0].read(index=index+1, force_type=np)
        self.cache = (index, current)
        flow = np.zeros_like(current,dtype=np.float32,shape=(*current.shape[0:2],2))
        for i in range(current.shape[2]):
            flow += cv.calcOpticalFlowFarneback(previous[:,:,i], current[:,:,i],
                                           None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)
        flow *= 1 / current.shape[2]
        return VideoSupplier.convert(flow, force_type)