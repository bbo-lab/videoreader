import numpy as np
from svidreader.video_supplier import VideoSupplier

import sys
sys.path.append("/home/paul/git/frame-interpolation")  # adjust path


class FILMInterpolation(VideoSupplier):
    def __init__(
        self,
        reader,
        model_path,
        device="cpu"
    ):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))

        import numpy as np
        import tensorflow as tf
        from inference.model_inference import ModelInference

        self.model = ModelInference(model_path, None)  # FILM uses TF internally
        self.cache = (None, None, None)

    def _preprocess(self, img):
        """
        FILM expects uint8 RGB images.
        Ensure correct dtype/shape.
        """
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _infer(self, img1, img2, t):
        """
        Direct interpolation (NO flow, NO warp).
        """
        return self.model.infer(img1, img2, t)

    def read(self, index, force_type=np):
        import numpy as np

        # exact frame
        if float(index).is_integer():
            return self.inputs[0].read(int(index), force_type=force_type)

        i0 = int(np.floor(index))
        alpha = float(index - i0)

        cache = self.cache
        if cache[0] == i0:
            img1, img2 = cache[1], cache[2]
        else:
            img1 = self.inputs[0].read(i0, force_type=np)
            img2 = self.inputs[0].read(i0 + 1, force_type=np)
            self.cache = (i0, img1, img2)

        img1 = self._preprocess(img1)
        img2 = self._preprocess(img2)

        out = self._infer(img1, img2, alpha)

        return VideoSupplier.convert(out, force_type)