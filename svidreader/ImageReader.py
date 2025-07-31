from enum import Enum
from imageio import v2 as imageio
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


#Enum for possible image types, tif, zip, raw
class ImageType(Enum):
    RAW = "raw"
    ZIP = "zip"
    TIF = "tif"
    FOLDER = "folder"
    IMG = "img"




class ImageRange(VideoSupplier):
    def __init__(self, folder_file, keyframe=None):
        self.frames = None
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
                self.filetype = ImageType.ZIP
            elif folder_file.endswith('.raw'):
                self.rawfile = open(folder_file, 'rb')
                self.rawfile.seek(0, 2)  # Move the cursor to the end of the file
                file_size = self.rawfile.tell()
                self.rawfile.seek(0)
                chunk = self.rawfile.read(64*2**16) #64MB
                chunk = unpack_10bit_to_16bit_fast(chunk)
                self.width = probe_width(chunk, 2, 1024)
                chunk = chunk[:(len(chunk) // self.width) * self.width].reshape(-1, self.width)
                self.height = probe_height(chunk, self.width // 16, 1024)
                self.depth = 10
                self.frames = np.arange(file_size * 8 // (self.depth * self.width * self.height))
                self.filetype = ImageType.RAW
            elif is_image(folder_file):
                if folder_file.endswith('.tif'):
                    #self.imagefile = imageio.mimread(folder_file)
                    import tifffile
                    with tifffile.TiffFile(folder_file) as tif:
                        self.imagefile = []
                        for pagenumber, page in enumerate(tif.pages):
                            tags = page.tags
                            img = page.asarray()
                            if 'PageNumber' in tags:
                                pagenumber = tags['PageNumber'].value[0]
                            if 'XPosition' in tags:
                                value = tags['XPosition'].value
                                img = np.roll(img, int(round(value[0] / value[1])), axis=1)
                            if 'YPosition' in tags:
                                value = tags['YPosition'].value
                                img = np.roll(img, int(round(value[0] / value[1])), axis=0)
                            while len(self.imagefile) <= pagenumber:
                                self.imagefile.append(None)
                            self.imagefile[pagenumber] = img

                    self.filetype = ImageType.TIF
                    super().__init__(n_frames=len(self.imagefile), inputs=())
                else:
                    super().__init__(n_frames=10000000, inputs=())
                    self.imagefile = imageio.imread(folder_file)
                    self.filetype = ImageType.IMG
            else:
                raise Exception(f"File ending of {folder_file} not understood")
        elif os.path.isdir(folder_file):
            files = os.listdir(folder_file)
            self.filetype = ImageType.FOLDER
        else:
            raise FileNotFoundError(f"Path {folder_file} does not Exist")
        if files is not None:
            files = np.sort(files)
            self.frames = []
            for f in files:
                if is_image(f):
                    self.frames.append(f"{folder_file}/{f}" if self.zipfile is None else f)
                elif f == "info.yml":
                    if self.zipfile is not None:
                        buf = self.zipfile.read(f)
                        fileinfo = yaml.safe_load(buf)
                        if keyframe is None:
                            self.keyframe = fileinfo.get("keyframe", self.keyframe)
        if self.frames is not None:
            super().__init__(n_frames=len(self.frames), inputs=())

    def read_impl(self, index):
        if self.imagefile is not None:
            if self.filetype == ImageType.TIF:
                return self.imagefile[index]
            return self.imagefile
        if self.rawfile is not None:
            framesize = (self.width * self.height * self.depth) // 8
            with self.mutex:
                self.rawfile.seek(framesize * index)
                chunk = self.rawfile.read(framesize)
            chunk = unpack_10bit_to_16bit_fast(chunk)
            return chunk.reshape(self.height, self.width, 1)
        if self.zipfile is not None:
            if not isinstance(index, str):
                frame_name = self.frames[index]
            else:
                frame_name = index
            try:
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                import cv2
                with self.mutex:
                    info = self.zipfile.getinfo(frame_name)
                    buf = self.zipfile.read(info)
                if frame_name.endswith("svg"):
                    return buf.decode("utf-8")
                np_buf = np.frombuffer(buf, np.uint8)
                res = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
                if res.ndim == 3 and res.shape[2] == 3:
                    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                return res
            except Exception as e:
                raise zipfile.BadZipFile(f"Cannot read file {self.folder_file}") from e
        return imageio.imread(self.frames[index])

    def read(self, index, force_type=np):
        res = self.read_impl(index)
        if isinstance(res, str):
            return res
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
    return ".png", ".exr", ".jpg", ".bmp", ".svg", "tif"
