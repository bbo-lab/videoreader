import logging

from svidreader.video_supplier import VideoSupplier
import numpy as np

logger = logging.getLogger(__name__)

# Module-level cv2 flag
cv2 = None  # None = not tried yet, False = import failed, module = imported

def shrink_by_integer_factor(img: np.ndarray, factor: int, same_output_type=True, xp=np):
    """
    Downscale an image by an integer factor using block averaging.
    Uses OpenCV INTER_AREA if available and dtype is supported (uint8, uint16, float32).
    """
    global cv2  # refer to the module-level flag

    if factor == 1:
        return img

    dtype = img.dtype
    supported_dtypes = (np.uint8, np.uint16, np.float32)

    # Try using OpenCV if dtype is supported
    if dtype in supported_dtypes:
        if cv2 is None:
            try:
                import cv2 as _cv2
                cv2 = _cv2
            except ImportError:
                cv2 = False  # mark as unavailable
                logger.log(logging.WARNING, "OpenCV not available, falling back to NumPy block-mean for downsampling.")

        if cv2:
            h, w = img.shape[:2]
            out_h = h // factor
            out_w = w // factor
            img_small = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if same_output_type and np.issubdtype(dtype, np.integer):
                img_small = xp.round(img_small).astype(dtype)
            return img_small

    # Fallback: NumPy block-mean
    img = img[:img.shape[0]//factor*factor, :img.shape[1]//factor*factor, ...]
    img = img.reshape(
        img.shape[0] // factor, factor,
        img.shape[1] // factor, factor,
        *img.shape[2:]
    )

    if xp.issubdtype(dtype, xp.integer):
        img = img.astype(xp.float32)
    img = xp.mean(img, axis=(1, 3))
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
