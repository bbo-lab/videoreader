import imageio
import numpy as np
from svidreader.video_supplier import VideoSupplier
import os
import zipfile
import yaml
from threading import Lock


def probe_width(data, min_width=2, max_width=None):
    """
    Probes possible widths and finds the one with maximum row similarity.

    Args:
        data (np.ndarray): 1D array of image data.
        min_width (int): Minimum width to try.
        max_width (int): Maximum width to try (default is len(data) // 2).

    Returns:
        int: Best width with maximum row similarity.
    """
    data = data.astype(np.int16)
    if max_width is None:
        max_width = len(data) // 2

    similarity = [np.inf] * min_width

    for width in range(min_width, max_width + 1):
        rows = data[:(len(data) // width) * width].reshape(-1, width)
        similarity.append(np.mean(np.abs(rows[1:] - rows[:-1])))

    return np.argmin(similarity)


def probe_height(data, min_height=2, max_height=None):
    """
    Probes possible heights and finds the one with maximum inter-image similarity.

    Args:
        data (np.ndarray): 1D array of image data.
        width (int): Width to use when reshaping into images.
        min_height (int): Minimum height to try.
        max_height (int): Maximum height to try (default is len(data) // width).

    Returns:
        int: Best height with maximum inter-image similarity.
    """
    data = data.astype(np.int16)
    max_height = max_height or data.shape[0]

    similarity = [np.inf] * min_height

    for height in range(min_height, max_height + 1):
        images = data[:(data.shape[0] // height) * height].reshape(-1, height, data.shape[1])
        similarity.append(np.mean(np.abs(images[1:]-images[:-1])))

    return  np.argmin(similarity)

def unpack_10bit_to_16bit_fast(packed_data):
    """
    Convert packed 10-bit integers to 16-bit integers using NumPy for better performance.

    Args:
        packed_data (bytes): The binary data containing packed 10-bit integers.

    Returns:
        np.ndarray: A NumPy array of 16-bit integers.
    """
    byte_array = np.frombuffer(packed_data, dtype=np.uint8)

    num_bits = len(byte_array) * 8
    aligned_bits = (num_bits // 10) * 10  # Align bits to multiples of 10
    packed_bits = np.unpackbits(byte_array, bitorder='little')[:aligned_bits]

    packed_bits = packed_bits.reshape(-1, 10)

    unpacked = np.packbits(packed_bits, axis=-1, bitorder='little').view(np.uint16)

    return unpacked

class ImageRange(VideoSupplier):
    def __init__(self, folder_file, keyframe=None):
        self.frames = []
        self.keyframe = keyframe
        self.zipfile = None
        self.imagefile = None
        self.rawfile = None
        self.width = None
        self.height = None
        self.depth = None
        files = None

        self.mutex = Lock()
        if os.path.isfile(folder_file):
            if folder_file.endswith('.zip'):
                try:
                    self.folder_file = folder_file
                    self.zipfile = zipfile.ZipFile(folder_file, "r")
                    files = self.zipfile.namelist()
                except Exception as e:
                    raise zipfile.BadZipFile(f"Cannot read file {self.folder_file}") from e
            elif folder_file.endswith('.raw'):
                self.rawfile = open(folder_file, 'rb')
                filesize = self.rawfile.tell()
                chunk = self.rawfile.read(64*2**16) #64MB
                chunk = unpack_10bit_to_16bit_fast(chunk)
                self.width = probe_width(chunk, 2, 1024)
                chunk = chunk[:(len(chunk) // self.width) * self.width].reshape(-1, self.width)
                self.height = probe_height(chunk, self.width // 16, 1024)
                self.depth = 10
                self.frames = np.arange(filesize * 8 // (self.depth * self.width * self.height))
            elif is_image(folder_file):
                super().__init__(n_frames=10000000, inputs=())
                self.imagefile = imageio.v2.imread(folder_file)
            else:
                raise Exception(f"File ending of {folder_file} not understood")
        if os.path.isdir(folder_file):
            files = os.listdir(folder_file)
        if files is not None:
            files = np.sort(files)
            for f in files:
                if is_image(f):
                    self.frames.append(f"{folder_file}/{f}" if self.zipfile is None else f)
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
        if self.rawfile is not None:
            framesize = (self.width * self.height * self.depth) // 8
            self.rawfile.seek(framesize * index)
            chunk = self.rawfile.read(framesize)
            chunk = unpack_10bit_to_16bit_fast(chunk)
            return chunk.reshape(self.height, self.width, 1)
        if self.zipfile is not None:
            frame_name = self.frames[index]
            try:
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
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
    imageEndings = get_image_endings()
    for ie in imageEndings:
        if filename.endswith(ie):
            return True
    return False

def get_image_endings():
    return ".png", ".exr", ".jpg", ".bmp"
