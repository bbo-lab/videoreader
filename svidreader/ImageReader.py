import imageio
import numpy as np
from svidreader.video_supplier import VideoSupplier
import os


class ImageRange(VideoSupplier):
    def __init__(self, folder_file, ncols=-1):
        self.frames = []
        imageEndings = get_imageEndings()

        if os.path.isdir(folder_file):
            for f in np.sort(os.listdir(folder_file)):
                if f.endswith(imageEndings):
                    self.frames.append(folder_file + "/" + f)
        else:
            self.frames.append(folder_file)

        super().__init__(n_frames=len(self.frames), inputs=())
        self.ncols = ncols

    def read(self, index, force_type=np):
        res = imageio.imread(self.frames[index])
        if res.ndim == 2:
            res = res[:, :, np.newaxis]
        return VideoSupplier.convert(res, force_type)

    def get_key_indices(self):
        return None


def get_imageEndings():
    return ".png", ".exr", ".jpg", ".bmp"
