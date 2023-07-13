import hashlib
import imageio.v3 as iio
from svidreader.imagecache import ImageCache
from ccvtools import rawio
import numpy as np

class VideoSupplier:
    def __init__(self, n_frames):
        self.n_frames = n_frames

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_frames

    def __enter__(self):
        return self

    def __next__(self):
        if (self.frame_idx + 1) < self.n_frames:
            self.frame_idx += 1
            return self.read(self.frame_idx)
        else:
            print("Reached end")
            raise StopIteration


class DumpToFile(VideoSupplier):
    def __init__(self, reader, output):
        self.reader = reader
        self.output = open(output, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self.reader.close()
        self.output.close()

    def read(self, index):
        data = self.reader.read(index=index)
        self.output.write(str(index) + ' ' + ' '.join(map(str, data)) + '\n')
        return data

    def improps(self):
        return self.reader.improps()

    def get_meta_data(self):
        return self.reader.get_meta_data()


class AnalyzeContrast(VideoSupplier):
    def __init__(self, reader):
        super().__init__(reader.n_frames * 3)
        self.reader = reader

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self.reader.close()

    def read(self, index):
        img = self.reader.read(index=index)
        gy, gx = np.gradient(img, axis=(0, 1))
        np.square(gx, out=gx)
        np.square(gy, out=gy)
        gx += gy
        np.sqrt(gx, out=gx)
        return np.average(gx)

    def improps(self):
        return self.reader.improps()

    def get_meta_data(self):
        return self.reader.get_meta_data()


class PermutateFrames(VideoSupplier):
    def __init__(self, reader, permutation):
        if isinstance(permutation, str):
            pass
        super().__init__(n_frames=len(permutation))
        self.reader = reader

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self.reader.close()

    def read(self, index):
        img = self.reader.read(index=permutation(index))
        return img

    def improps(self):
        return self.reader.improps()

    def get_meta_data(self):
        return self.reader.get_meta_data()


class BgrToGray(VideoSupplier):
    def __init__(self, reader):
        super().__init__(reader.n_frames * 3)
        self.reader = reader

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self.reader.close()

    def read(self, index):
        img = self.reader.read(index=index // 3)
        return img[:,:,index % 3]

    def improps(self):
        return self.reader.improps()

    def get_meta_data(self):
        return self.reader.get_meta_data()


class FrameDifference(VideoSupplier):
    def __init__(self, reader):
        super().__init__(reader.n_frames - 1)
        self.reader = reader

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self.reader.close()

    def read(self, index):
        return 128 + self.reader.read(index=index + 1) - self.reader.read(index=index)

    def improps(self):
        return self.reader.improps()

    def get_meta_data(self):
        return self.reader.get_meta_data()
