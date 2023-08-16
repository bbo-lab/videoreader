import hashlib
import imageio.v3 as iio
from svidreader.video_supplier import VideoSupplier
from ccvtools import rawio
import numpy as np

class DumpToFile(VideoSupplier):
    def __init__(self, reader, outputfile):
        import imageio
        super().__init__(n_frames= reader.n_frames, inputs=(reader,))
        self.outputfile = outputfile
        if outputfile.endswith('.mp4'):
            self.type = "movie"
            self.output =  imageio.get_writer(outputfile)
        else:
            self.type = "csv"
            self.output = open(outputfile, 'w')

    def close(self):
        super().close()
        self.output.close()

    def read(self, index):
        data = self.inputs[0].read(index=index)
        if self.type == "movie":
            self.output.append_data(data)
        elif self.type == "csv":
            self.output.write(str(index) + ' ' + ' '.join(map(str, data)) + '\n')
        return data

class Arange(VideoSupplier):
    def __init__(self, inputs, ncols=-1):
        super().__init__(n_frames=inputs[0].n_frames, inputs=inputs)
        self.ncols = ncols

    def read(self, index):
        grid = [[]]
        maxdim = np.zeros(shape=(3,), dtype=int)
        for r in self.inputs:
            if len(grid[-1]) == self.ncols:
                grid.append([])
            img = r.read(index=index)
            grid[-1].append(img)
            maxdim = np.maximum(maxdim, img.shape)
        res = np.zeros(shape=(maxdim[0] * len(grid), maxdim[1] * len(grid[0]), maxdim[2]),dtype=grid[0][0].dtype)
        for col in range(len(grid)):
            for row in range(len(grid[col])):
                img = grid[col][row]
                res[col * maxdim[0]: col * maxdim[0] + img.shape[0], row * maxdim[1]: row * maxdim[1] + img.shape[1]] = img
        return res

class Crop(VideoSupplier):
    def __init__(self, reader, width=-1, height=-1):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.width = width
        self.height = height

    def read(self, index):
        img = self.inputs[0].read(index=index)
        return img[0:self.height, 0:self.width]

class AnalyzeContrast(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))

    def read(self, index):
        img = self.inputs[0].read(index=index)
        gy, gx = np.gradient(img, axis=(0, 1))
        np.square(gx, out=gx)
        np.square(gy, out=gy)
        gx += gy
        np.sqrt(gx, out=gx)
        return np.average(gx)

class Scale(VideoSupplier):
    def __init__(self, reader, scale):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.scale = scale

    def read(self, index):
        import cv2
        img = self.inputs[0].read(index=index)
        resized = cv2.resize(img, (int(img.shape[1] * self.scale), int(img.shape[0] * self.scale)))
        return resized

def read_numbers(filename):
    with open(filename, 'r') as f:
        return np.asarray([int(x) for x in f],dtype=int)


class TimeToFrame(VideoSupplier):
    def __init__(self, reader, timingfile):
        import pandas
        timings = pd.read_csv(timingfile)


class PermutateFrames(VideoSupplier):
    def __init__(self, reader, permutation):
        if isinstance(permutation, str):
            permutation = read_numbers(permutation)
        super().__init__(n_frames=len(permutation), inputs=(reader,))

    def read(self, index):
        img = self.inputs[0].read(index=permutation[index])
        return img

class BgrToGray(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames * 3, inputs=(reader,))

    def read(self, index):
        img = self.inputs[0].read(index=index // 3)
        return img[:,:,[index % 3]]


class FrameDifference(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames - 1, inputs = (reader,))

    def read(self, index):
        return 128 + self.inputs[0].read(index=index + 1) - self.inputs[0].read(index=index)
