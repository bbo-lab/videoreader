from svidreader.video_supplier import VideoSupplier
import numpy as np

def shrink_by_integer_factor(img:np.ndarray, factor:int, same_output_type=True, xp=np):
    if factor == 1:
        return img
    dtype = img.dtype
    img = img.reshape(img.shape[0] // factor, factor, img.shape[1] // factor, factor, *img.shape[2:])
    img = xp.average(img, axis=(1, 3))
    if same_output_type:
        if xp.issubdtype(dtype, xp.integer):
            img = xp.round(img)
        img = img.astype(dtype)
    return img

class DownsampleResolution(VideoSupplier):
    def __init__(self, reader, factor):
        super().__init__(n_frames=(reader.n_frames), inputs=(reader,))
        self.factor = factor

    def read(self, index, force_type=None):
        frame = self.inputs[0].read(index, force_type=force_type)
        if self.factor == 1:
            return frame
        return shrink_by_integer_factor(frame, self.factor)

    def prefetch(self, index):
        return self.inputs[0].prefetch(index)