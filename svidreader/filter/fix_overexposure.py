
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
import logging

from svidreader.video_supplier import VideoSupplier

logger = logging.getLogger(__name__)

class FixOverexposure(VideoSupplier):
    def __init__(
        self,
        reader,
        overexposure_boost=1.0,
        max_value=None,
        rtol=1e-6,
        maxiter=None,
    ):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.mode = "distance"  # Default mode

        self.overexposure_boost = int(overexposure_boost)
        self.max_value = int(max_value)
        self.rtol = rtol
        self.maxiter = maxiter

        if self.overexposure_boost <= 0:
            raise ValueError("overexposure_boost must be positive")

    def read(self, index, force_type=None):
        frame = self.inputs[0].read(index=index, force_type=force_type)

        if frame.ndim not in (2, 3):
            raise ValueError("Expected a grayscale or HxWxC image")

        if self.max_value is not None:
            max_value = self.max_value

        # Solve in floating point; preserve the input shape.
        original_dtype = frame.dtype
        image = frame.astype(np.float32, copy=True)

        for c in range(image.shape[2]):
            if self.mode == "distance":
                image[..., c] = self._fix_channel_distance(
                    image[..., c], max_value
                )
            elif self.mode == "quadratic":
                image[..., c] = self._fix_channel_quadratic(
                    image[..., c], max_value
                )
            else:
                raise ValueError(
                    f"Unknown mode {self.mode}, expected 'distance' or 'quadratic'"
                )

        # Preserve the original image's numeric range and dtype.
        if np.issubdtype(original_dtype, np.integer):
            limits = np.iinfo(original_dtype)
            max_value = image.max()
            if max_value > limits.max:
                logger.log(logging.WARNING, f"FixOverexposure: max value {max_value} exceeds limits of {limits.max}, clipping to limits")
                image = image * (limits.max / max_value)
            image = np.clip(image, limits.min, limits.max)
            image = np.rint(image)

        image = image.astype(original_dtype, copy=False)

        #print how much the boosting increased the max value of the image in percentage (0% means no change, 100% means doubled)
        if self.max_value is not None:
            new_max_value = image.max()
            if new_max_value > self.max_value:
                increase_percentage = (new_max_value - self.max_value) / self.max_value * 100
                logger.log(logging.INFO, f"FixOverexposure: max value increased by {increase_percentage:.2f}%")

        return force_type.asarray(image) if force_type is not None else image

    def _fix_channel_distance(self, image, max_value):
        mask = image >= max_value

        if not mask.any() or mask.all():
            return image

        # Distance from each masked pixel to the nearest non-masked pixel.
        module = VideoSupplier.get_array_module_name(image)
        if module == "numpy":
            import scipy.ndimage as ndimage
        elif module == "cupy":
            import cupyx.scipy.ndimage as ndimage
        distance = ndimage.distance_transform_edt(mask)
        result = image.copy()
        result[mask] = max_value + self.overexposure_boost * distance[mask]
        return result


    def _fix_channel_quadratic(self, image, max_value):
        height, width = image.shape
        mask = image >= max_value
        #erode mask by one pixel
        mask = mask & np.roll(mask, 1, axis=0) & np.roll(mask, -1, axis=0) & np.roll(mask, 1, axis=1) & np.roll(mask, -1, axis=1)
        num_masked = np.count_nonzero(mask)

        if num_masked == 0 or num_masked == image.size:
            return image


        # Map unknown pixels to consecutive matrix indices.
        ids = np.full((height, width), -1, dtype=np.int32)
        ids[mask] = np.arange(num_masked, dtype=np.int32)

        # A is the graph Laplacian on the unknown pixels.
        diagonal = np.zeros(num_masked, dtype=np.float32)
        b = np.full(num_masked, self.overexposure_boost, dtype=np.float32)

        rows = []
        cols = []
        values = []

        # Four-connected 2D neighborhood.
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r0 = max(0, -dy)
            r1 = min(height, height - dy)
            c0 = max(0, -dx)
            c1 = min(width, width - dx)

            rr, cc = np.nonzero(mask[r0:r1, c0:c1])
            rr = rr + r0
            cc = cc + c0

            nr = rr + dy
            nc = cc + dx

            src = ids[rr, cc]
            dst = ids[nr, nc]

            # Every valid neighbor contributes to the diagonal.
            diagonal[src] += 1.0

            unknown_neighbor = dst >= 0

            # Unknown neighbors produce off-diagonal -1 entries.
            rows.append(src[unknown_neighbor])
            cols.append(dst[unknown_neighbor])
            values.append(
                np.full(unknown_neighbor.sum(), -1.0, dtype=np.float32)
            )

            # Known neighbors contribute their values to the RHS.
            known_neighbor = ~unknown_neighbor
            b[src[known_neighbor]] += image[
                nr[known_neighbor], nc[known_neighbor]
            ]

        # Add the diagonal entries.
        rows.append(np.arange(num_masked, dtype=np.int32))
        cols.append(np.arange(num_masked, dtype=np.int32))
        values.append(diagonal)

        A = sp.coo_matrix(
            (
                np.concatenate(values),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
            shape=(num_masked, num_masked),
            dtype=np.float32,
        ).tocsr()

        x, info = cg(A, b, rtol=self.rtol, maxiter=self.maxiter)

        if info != 0:
            raise RuntimeError(
                f"Conjugate gradient did not converge (info={info})"
            )

        result = image.copy()
        result[mask] = x
        return result
