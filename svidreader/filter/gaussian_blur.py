from svidreader.video_supplier import VideoSupplier
import numpy as np
import logging
import inspect

logger = logging.getLogger(__name__)

class GaussianBlur(VideoSupplier):
    fallback = False

    def __init__(self, reader, sigma=1):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.sigma = sigma

    def read(self, index, force_type=None):
        img = self.inputs[0].read(index=index, force_type=force_type)
        xp = inspect.getmodule(type(img))

        if xp == np:
            from scipy.ndimage import gaussian_filter
            result = gaussian_filter(img, sigma=self.sigma, axes=(0, 1))
        else:
            if not GaussianBlur.fallback:
                from cupy.cuda.compiler import CompileException
                try:
                    from cupyx.scipy.ndimage import gaussian_filter
                    result = xp.stack([gaussian_filter(img[..., i], sigma=self.sigma) for i in range(img.shape[2])], axis=-1)
                except CompileException as e:
                    logger.log(logging.WARNING, "Falling back to numpy for Gaussian blur due to compilation error:", e)
                    GaussianBlur.fallback = True
            if GaussianBlur.fallback:
                from scipy.ndimage import gaussian_filter
                img = VideoSupplier.convert(img, np)
                result_np = gaussian_filter(img, sigma=self.sigma, axes=(0, 1))
                result = VideoSupplier.convert(result_np, force_type)
        return result