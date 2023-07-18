from svidreader.video_supplier import VideoSupplier
from scipy.interpolate import RegularGridInterpolator
import yaml
import numpy as np
from calibcamlib import Camerasystem

class PerspectiveCameraProjection(VideoSupplier):
    def __init__(self, reader, config_file):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        with open(config_file, 'r') as f:
            eye_video_config = yaml.safe_load(f)
        self.cam_calibs = calib_dicts[camera_ind]['calibs']
        self.cam_calibs[0]['rvec_cam'] = np.zeros_like(cam_calibs[0]['rvec_cam'])
        self.cam_calibs[0]['tvec_cam'] = np.zeros_like(cam_calibs[0]['tvec_cam'])

        self.cs = Camerasystem.from_calibs(np.load(eye_video_config['calibfile'])['calibs'])

        center = calc_center(calibs)
        effective_image_radius = calc_effective_radius(calibs)

        sca_fac = eye_video_config["perspective_video"]["scaling_factor"]
        h = eye_video_config["perspective_video"]["height"]
        w = eye_video_config["perspective_video"]["width"]

        # The view vectors that defines the subimage
        view_long = eye_video_config["view_long"]
        view_lat = eye_video_config["view_lat"]

        self.orig_object_points = get_object_points(sca_fac, h, w)
        self.image_points = None

    # Function to generate object points with defined scaling factor and image dimensions
    @staticmethod
    def get_object_points(focal_length, image_h, image_w):
        a = np.arange(image_w) - (image_w / 2)
        b = np.arange(image_h) - (image_h / 2)

        a_image, b_image = np.meshgrid(a, b)

        x, y, z = (a_image.flatten(), b_image.flatten(), focal_length * np.ones_like(a_image).flatten())
        return np.vstack([x, y, z]).T

    def get_data(self, index):
        frame = self.inputs[0].read(index=index)
        if self.image_points is None:
            self.image_points, mask = calc_perspective_map_with_calibs(calibs=self.cam_calibs,
                                                              object_points=self.final_points,
                                                              orig_image_shape=frame.shape)

        interpolater = RegularGridInterpolator((np.arange(frame.shape[0]),
                                                np.arange(frame.shape[1])),
                                                frame)
        perspective_image = interpolater(image_points,
                                         method=eye_video_config["output_video"]["interpolate"]["method"])