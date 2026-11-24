from svidreader.video_supplier import VideoSupplier
import numpy as np


class MapCoordinates(VideoSupplier):
    """
    Remap video frames using per-pixel image coordinates.

    Parameters
    ----------
    reader : VideoSupplier
        Source video supplier.
    image_points : np.ndarray or VideoSupplier
        Coordinates used to sample the input frame. A static array must have
        shape ``(height, width, 2)``. A per-frame VideoSupplier should provide
        coordinates with shape ``(height, width, 2)``.
    interpolation_order : int, default=1
        Spline interpolation order passed to ``scipy.ndimage.map_coordinates``
        (or its CuPy equivalent).

    Returns
    -------
    np.ndarray or cupy.ndarray
        The remapped frame with shape ``(height, width, channels)``. The output type follows ``force_type`` (NumPy or
        CuPy).

    Notes
    -----
    Coordinates are interpreted as ``(row, column)`` sampling coordinates.
    Values outside the source image are filled with the background color.
    """
    def __init__(
            self,
            reader:VideoSupplier,
            image_points:np.ndarray|VideoSupplier,
            interpolation_order=1,
            mode='constant',
            background_color=(0, 0, 0)):
        inputs = (reader,image_points) if isinstance(image_points, VideoSupplier) else (reader,)
        super().__init__(n_frames=reader.n_frames, inputs=inputs)
        self.interpolation_order = interpolation_order
        self.image_to_int = {}
        self.mode = mode
        self.background_color = background_color
        if isinstance (image_points, np.ndarray):
            assert image_points.ndim > 2, f"image_points must be a 3D array, got {image_points.ndim}D array"
            assert image_points.shape[-1] == 2, f"image_points must have shape (height, width, 2), got {image_points.shape}"
        self.image_points = image_points if isinstance(image_points, VideoSupplier) else np.moveaxis(image_points, -1, -3)

    def get_image_float_to_int(self, xp):
        res = self.image_to_int.get(xp, None)
        if res is None:
            def image_float_to_int(image, xp=xp):
                image = xp.round(image)
                image = xp.clip(image, 0, 255)
                return image.astype(xp.uint8)
            if xp == np:
                try:
                    import numba as nb
                    res = nb.njit(image_float_to_int)
                except ImportError:
                    res = image_float_to_int
            else:
                res = xp.fuse(image_float_to_int)

            self.image_to_int[xp] = res
        return res

    def read(self, index, force_type=None):
        if force_type is None:
            force_type = np
        match force_type.__name__:
            case "numpy":
                from scipy.ndimage import map_coordinates
            case "cupy":
                from cupyx.scipy.ndimage import map_coordinates
            case _:
                raise Exception(f"--use-lib={force_type} must be one of cupy,numpy")


        frame = self.inputs[0].read(index, force_type=force_type)
        original_dtype = frame.dtype

        frame = frame.astype(force_type.float32, copy=False)
        frame = VideoSupplier.convert(frame, force_type)
        image_points = self.image_points
        if isinstance(image_points, VideoSupplier):
            image_points = image_points.read(index, force_type=force_type)
            image_points = np.moveaxis(image_points, 2, 0)
        elif isinstance(image_points, np.ndarray):
            if image_points.ndim == 4:
                image_points = image_points[index]
        image_points = VideoSupplier.convert(image_points, force_type)
        assert image_points.shape[0] == 2, f"image_points must have shape (2, height, width), got {image_points.shape}"
        perspective_image = [
            map_coordinates(frame[:, :, i], image_points, mode=self.mode, cval=self.background_color[i], order=self.interpolation_order) for i in
            range(frame.shape[2])]

        perspective_image = force_type.stack(perspective_image, axis=2)
        if original_dtype == np.uint8:
            perspective_image = self.get_image_float_to_int(xp=force_type)(perspective_image)
        return perspective_image