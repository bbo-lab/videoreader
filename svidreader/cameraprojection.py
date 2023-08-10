from svidreader.video_supplier import VideoSupplier
from scipy.interpolate import RegularGridInterpolator
import yaml
import numpy as np
from calibcamlib import Camerasystem
from scipy.spatial.transform import Rotation


class PerspectiveCameraProjection(VideoSupplier):
    def __init__(self, reader, config_file=None):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))

        # Necessary
        self.image_points = None
        self.method = None
        self.h = None
        self.w = None

        # Misc
        self.perspective_config = None

        if config_file is not None:
            with open(config_file, 'r') as f:
                self.perspective_config = yaml.safe_load(f)

            cam_calibs = np.load(self.perspective_config['calib_file'], allow_pickle=True).item()['calibs']
            cam_calibs[0]['rvec_cam'] = np.zeros_like(cam_calibs[0]['rvec_cam'])
            cam_calibs[0]['tvec_cam'] = np.zeros_like(cam_calibs[0]['tvec_cam'])
            cs = Camerasystem.from_calibs(cam_calibs)

            sca_fac = self.perspective_config["perspective_video"]["scaling_factor"]
            self.h = self.perspective_config["perspective_video"]["height"]
            self.w = self.perspective_config["perspective_video"]["width"]
            orig_object_points = self.get_object_points(sca_fac, self.h, self.w)

            # The view vectors that defines the subimage
            view_long = self.perspective_config["view_long"]
            view_lat = self.perspective_config["view_lat"]
            final_points = Rotation.from_euler('xyz', [-view_lat, view_long, 0], degrees=True).apply(orig_object_points)
            self.method = self.perspective_config["output_video"]["interpolate"]["method"]
            self.image_points = cs.project(final_points)[0]
            self.image_points = self.image_points.squeeze()[:, ::-1]

    @staticmethod
    def get_object_points(focal_length, image_h, image_w):
        """Function to generate object points with defined scaling factor and image dimensions"""
        a = np.arange(image_w) + (0.5 - image_w / 2)
        b = np.arange(image_h) + (0.5 - image_h / 2)

        a_image, b_image = np.meshgrid(a, b)

        x, y, z = (a_image.flatten(), b_image.flatten(), focal_length * np.ones_like(a_image).flatten())
        points = np.vstack([x, y, z]).T
        norm = np.linalg.norm(points, axis=-1, keepdims=True)
        return np.where(norm == 0, points, points / norm)

    def read(self, index):
        frame = self.inputs[0].read(index=index)

        if self.image_points is None:
            return frame

        interpolater = RegularGridInterpolator((np.arange(frame.shape[0]),
                                                np.arange(frame.shape[1])),
                                                frame,
                                                bounds_error=False,
                                                fill_value=0.0)
        perspective_image = interpolater(self.image_points, method=self.method)
        perspective_image = perspective_image.reshape(self.h, self.w, -1)
        np.round(perspective_image, out=perspective_image)
        np.clip(perspective_image, 0, 255, out=perspective_image)
        return perspective_image.astype(np.uint8)

    def get_scaling_factor(self):
        if self.perspective_config is not None:
            return self.perspective_config["perspective_video"]["scaling_factor"]
        else:
            return None
