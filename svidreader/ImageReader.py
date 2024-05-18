import imageio
import numpy as np
from svidreader.video_supplier import VideoSupplier
import os
import zipfile


class ImageRange(VideoSupplier):
    def __init__(self, folder_file, ncols=-1, keyframe=None):
        self.frames = []
        self.keyframe = keyframe
        imageEndings = get_imageEndings()
        self.zipfile = None
        if os.path.isfile(folder_file) and folder_file.endswith('.zip'):
            self.zipfile = zipfile.ZipFile(folder_file, "r")
            files = self.zipfile.namelist()
        else:
            files = os.listdir(folder_file)
        files = np.sort(files)
        for f in files:
            for ie in imageEndings:
                if f.endswith(ie):
                    self.frames.append(folder_file + "/" + f if self.zipfile is None else f)
                    break
            if f == "info.yml":
                if self.zipfile is not None:
                    buf = self.zipfile.read(self.frames[index])
                    self.fileinfo = yaml.safe_load(buf)
                    if keyframe is None:
                        self.keyframe = fileinfo.get("keyframe", self.keyframe)
        super().__init__(n_frames=len(self.frames), inputs=())
        self.ncols = ncols

    def read_impl(self, index):
        if self.zipfile is not None:
            import cv2
            buf = self.zipfile.read(self.frames[index])
            np_buf = np.frombuffer(buf, np.uint8)
            res = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
            if res.ndim == 3 and res.shape[2] == 3:
                res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            return res
        return imageio.imread(self.frames[index])

    def read(self, index, force_type=np):
        res = self.read_impl(index)
        if self.keyframe is not None:
            if index % self.keyframe != 0:
                res += self.read_impl((index // self.keyframe) * self.keyframe)
                res += 129
        if res.ndim == 2:
            res = res[:, :, np.newaxis]
        return VideoSupplier.convert(res, force_type)

    def get_key_indices(self):
        return None


def get_imageEndings():
    return ".png", ".exr", ".jpg", ".bmp"
