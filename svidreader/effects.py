import hashlib
import imageio.v3 as iio
from svidreader.video_supplier import VideoSupplier
from ccvtools import rawio
import numpy as np

class DumpToFile(VideoSupplier):
    def __init__(self, reader, output):
        super().__init__(n_frames= reader.n_frames, inputs=(reader,))
        self.output = open(output, 'w')

    def close(self):
        super().close()
        self.output.close()

    def read(self, index):
        data = self.inputs[0].read(index=index)
        self.output.write(str(index) + ' ' + ' '.join(map(str, data)) + '\n')
        return data

class Arange(VideoSupplier):
    def __init__(self, reader, rows, cols):
        super().__init__(n_frames=reader[0].n_frames, inputs=[reader])

    def read(self, index):
        img = []
        for r in self.inputs:
            img.append(r.read(index=index))
        return img

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

def read_numbers(filename):
    with open(filename, 'r') as f:
        return np.asarray([int(x) for x in f],dtype=int)


class TimeToFrame(VideoSupplier):
    def __init__(self, reader, timingfile):
        import pandas


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
        return img[:,:,index % 3]


class FrameDifference(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames - 1, inputs = (reader,))

    def read(self, index):
        return 128 + self.inputs[0].read(index=index + 1) - self.inputs[0].read(index=index)
