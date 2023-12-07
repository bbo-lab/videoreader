import imageio
import numpy as np
from svidreader.video_supplier import VideoSupplier
import os

class ImageRange(VideoSupplier):
    def __init__(self, folder, ncols=-1):
        self.frames = []
        imageEndings = (".png",".exr",".jpg",".bmp")
        for f in np.sort(os.listdir(folder)):
            for ie in imageEndings:
                if f.endswith(imageEndings):
                    self.frames.append(folder + "/" + f)
                    break
        super().__init__(n_frames=len(self.frames), inputs=())
        self.ncols = ncols

    def read(self, index, force_type=np):
        return VideoSupplier.convert(imageio.imread(self.frames[index]), force_type)

    def get_key_indices(self):
        return None