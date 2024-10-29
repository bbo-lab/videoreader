from duplicity.config import force

from svidreader.video_supplier import VideoSupplier
import numpy as np


class MapCoordinates(VideoSupplier):
    def __init__(self, reader, image_points, interpolation_order=1):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.interpolation_order = interpolation_order
        self.image_to_int = {}
        self.image_points = np.moveaxis(image_points, 2, 0)

    def get_image_float_to_int(self, xp):
        res = self.image_to_int.get(xp, None)
        if res is not None:
            return res
        def image_float_to_int(image, xp=xp):
            image = xp.round(image)
            image = xp.clip(image, 0, 255)
            return image.astype(xp.uint8)
        if xp == np:
            import numba as nb
            res = nb.njit(image_float_to_int)
        else:
            res = xp.fuse(self.image_to_int)
        self.image_to_int[xp] = res
        return res

    def read(self, index, force_type=np):
        match force_type.__name__:
            case "numpy":
                from scipy.ndimage import map_coordinates
            case "cupy":
                from cupyx.scipy.ndimage import map_coordinates
            case _:
                raise Exception(f"--use-lib={force_type} must be one of cupy,numpy")


        frame = self.inputs[0].read(index, force_type=force_type)

        frame = frame.astype(force_type.float32)

        perspective_image = [
            map_coordinates(frame[:, :, i], self.image_points, mode='constant', cval=0, order=self.interpolation_order) for i in
            range(frame.shape[2])]
        perspective_image = force_type.stack(perspective_image, axis=2)
        return self.get_image_float_to_int(force_type)(perspective_image)