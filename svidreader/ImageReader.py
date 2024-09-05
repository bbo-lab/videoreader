import imageio
import numpy as np
from svidreader.video_supplier import VideoSupplier
import os
import zipfile
import yaml
from threading import Lock


class ImageRange(VideoSupplier):
    def __init__(self, folder_file, keyframe=None):
        self.frames = []
        self.keyframe = keyframe
        self.zipfile = None
        self.imagefile = None
        self.mutex = Lock()
        if os.path.isfile(folder_file) and folder_file.endswith('.zip'):
            try:
                self.folder_file = folder_file
                self.zipfile = zipfile.ZipFile(folder_file, "r")
                files = self.zipfile.namelist()
            except Exception as e:
                raise zipfile.BadZipFile(f"Cannot read file {self.folder_file}") from e
        elif os.path.isfile(folder_file) and is_image(folder_file):
            super().__init__(n_frames=10000000, inputs=())
            self.imagefile = imageio.v2.imread(folder_file)
            return
        else:
            files = os.listdir(folder_file)
        files = np.sort(files)
        for f in files:
            if is_image(f):
                self.frames.append(folder_file + "/" + f if self.zipfile is None else f)
            elif f == "info.yml":
                if self.zipfile is not None:
                    buf = self.zipfile.read(f)
                    fileinfo = yaml.safe_load(buf)
                    if keyframe is None:
                        self.keyframe = fileinfo.get("keyframe", self.keyframe)
        super().__init__(n_frames=len(self.frames), inputs=())

    def read_impl(self, index):
        if self.imagefile is not None:
            return self.imagefile
        if self.zipfile is not None:
            frame_name = self.frames[index]
            try:
                import cv2
                with self.mutex:
                    buf = self.zipfile.read(frame_name)
                np_buf = np.frombuffer(buf, np.uint8)
                res = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
                if res.ndim == 3 and res.shape[2] == 3:
                    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                return res
            except Exception as e:
                raise zipfile.BadZipFile(f"Cannot read file {self.folder_file}") from e
        return imageio.v2.imread(self.frames[index])

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
        return np.arange(0, self.n_frames)

    def __del__(self):
        super(ImageRange, self).__del__()
        if self.zipfile is not None:
            self.zipfile.close()
            self.zipfile = None

def is_image(filename):
    imageEndings = get_imageEndings()
    for ie in imageEndings:
        if filename.endswith(ie):
            return True
    return False

def get_imageEndings():
    return ".png", ".exr", ".jpg", ".bmp"
