import hashlib
import imageio.v3 as iio
from svidreader.imagecache import ImageCache
from ccvtools import rawio


class BgrToGray:
    def __init__(self, reader, n_frames=0):
        self.reader = reader
        self.n_frames = n_frames * 3

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self.reader.close()

    def get_data(self, fr_idx):
        img = self.reader.get_data(fr_idx // 3)
        return img[:,:,fr_idx % 3]

    def improps(self):
        return self.reader.improps()

    def get_meta_data(self):
        return self.reader.get_meta_data()

    def __iter__(self):
        return self

    def __next__(self):
        if (self.frame_idx + 1) < self.n_frames:
            self.frame_idx += 1
            return self.get_data(self.frame_idx)
        else:
            print("Reached end")
            raise StopIteration

    def __len__(self):
        return self.n_frames

class FrameDifference:
    def __init__(self, reader, n_frames=0):
        self.reader = reader
        self.n_frames = n_frames - 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self.reader.close()

    def get_data(self, fr_idx):
        return self.reader.get_data(fr_idx + 1) - self.reader.get_data(fr_idx)

    def improps(self):
        return self.reader.improps()

    def get_meta_data(self):
        return self.reader.get_meta_data()

    def __iter__(self):
        return self

    def __next__(self):
        if (self.frame_idx + 1) < self.n_frames:
            self.frame_idx += 1
            return self.get_data(self.frame_idx)
        else:
            print("Reached end")
            raise StopIteration

    def __len__(self):
        return self.n_frames
