from svidreader.video_supplier import VideoSupplier
import numpy as np
class BlinkingLightDetector(VideoSupplier):
    def __init__(self, reader, lbound, ):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.window = window
        self.scale = float(scale)
        self.cache = {}
        self.stack = {}
        self.foreground = foreground
        print(scale, window, foreground)

    @xp.fuse()
    def gauss(x, y, scale):
        diff = (x - y) * (1 / scale)
        return xp.exp(-xp.sum(xp.square(diff), axis=2))

    def read(self, index):
        curframe = np.convolve(self.inputs[0].read(index=index + 1).astype(np.int16))
        curfraame = np.convolve(curframe, [-1,-1,0,1,2,1, 0,-1,-1] , axis=0,mode='same')
        curfraame = np.convolve(curframe, [-1,-1,0,1,2,1, 0,-1,-1],  axis=1,mode='same')
        lastframe =self.inputs[0].read(index=index + 1).astype(np.int16)
        lastframe = np.convolve(lastframe, [1,2,1] , axis=0,mode='same')
        lastframe = np.convolve(lastframe, [1,2,1],  axis=1,mode='same')
        return curframe - lastframe