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
            self.mapkeys = None
            self.output = open(outputfile, 'w')

    def close(self):
        super().close()
        self.output.close()

    def read(self, index):
        data = self.inputs[0].read(index=index)
        if self.type == "movie":
            if data is not None:
                self.output.append_data(data)
        elif self.type == "csv":
            if self.mapkeys == None and isinstance(data, dict):
                self.mapkeys = data.keys()
                self.output.write("index " + ' '.join(self.mapkeys) + '\n')
            self.output.write(str(index) + ' ' + ' '.join([str(data[k]) for k in self.mapkeys]) + '\n')
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
    def __init__(self, reader, x = 0, y = 0, width=-1, height=-1):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def read(self, index):
        img = self.inputs[0].read(index=index)
        return img[self.x : self.x + self.height, self.y : self.y + self.width]


class Math(VideoSupplier):
    def __init__(self, reader, expression):
        super().__init__(n_frames=reader[0].n_frames, inputs=reader)
        print('expression',expression)
        self.exp = compile(expression, '<string>', 'exec')

    def read(self, index):
        args = {'i' + str(i) : self.inputs[i].read(index = index) for i in range(len(self.inputs))}
        args['np'] = np
        ldict = {}
        exec(self.exp, args, ldict)
        return ldict['out']

class AnalyzeImage(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))

    def read(self, index):
        img = self.inputs[0].read(index=index)
        gy, gx = np.gradient(img, axis=(0, 1))
        np.square(gx, out=gx)
        np.square(gy, out=gy)
        gx += gy
        np.sqrt(gx, out=gx)
        return {'contrast':np.average(gx), 'brightness':np.average(img)}


class MaxIndex(VideoSupplier):
    def __init__(self, reader, count, radius):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.count = int(count)
        self.radius = int(radius)

    def read(self, index):
        import cv2
        img = self.inputs[0].read(index=index)
        img = np.array(img,copy=True)
        i = 0
        res = {}
        while True:
            maxpix = np.argmax(img)
            maxpix = np.unravel_index(maxpix, img.shape[0:2])
            res['x'+str(i)] = maxpix[0]
            res['y'+str(i)] = maxpix[1]
            res['c'+str(i)] = img[maxpix[0],maxpix[1]]
            i = i + 1
            if i > self.count:
                break
            cv2.circle(img, (maxpix[1],maxpix[0]), self.radius, 0, -1)
        return res


class Plot(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, input=(reader,))

    def read(self, index):
        img = self.inputs[0].read(index=index)
        data = self.inputs[1].read(index=index)
        img = np.copy(img)
        cv2.circle(img, (data['x'], data['y']), 2, (255, 0, 0), data['c'])
        return img


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

def read_map(filename):
    res = {}
    import pandas as pd
    csv = pd.read_csv(filename, sep=' ')
    return dict(zip(csv['from'], csv['to']))


class TimeToFrame(VideoSupplier):
    def __init__(self, reader, timingfile):
        import pandas
        timings = pd.read_csv(timingfile)


class PermutateFrames(VideoSupplier):
    def __init__(self, reader, permutation=None, mapping=None):
        if isinstance(permutation, str):
            permutation = read_numbers(permutation)
        if isinstance(mapping, str):
            permutation = read_map(mapping)
        self.permutation = permutation
        self.invalid = np.zeros_like(reader.read(index=0))
        super().__init__(n_frames=len(permutation), inputs=(reader,))

    def read(self, index):
        if index in self.permutation:
            return self.inputs[0].read(index=self.permutation[index])
        else:
            return self.invalid

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
