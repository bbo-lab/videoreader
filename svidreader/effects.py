from svidreader.video_supplier import VideoSupplier
import numpy as np


class DumpToFile(VideoSupplier):
    def __init__(self, reader, outputfile, makedir=False, comment=None):
        import imageio
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.outputfile = outputfile
        if makedir:
            from pathlib import Path
            Path(outputfile).parent.mkdir(parents=True, exist_ok=True)
        if outputfile.endswith('.mp4'):
            self.type = "movie"
            self.output = imageio.get_writer(outputfile)
        else:
            self.type = "csv"
            self.mapkeys = None
            self.output = open(outputfile, 'w')
            if comment is not None:
                self.output.write(comment + '\n')

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
                self.output.write(f"index {' '.join(self.mapkeys)} \n")
            self.output.write(f"{index} {' '.join([str(data[k]) for k in self.mapkeys])} \n")
        return data


class Arange(VideoSupplier):
    def __init__(self, inputs, ncols=-1):
        super().__init__(n_frames=inputs[0].n_frames, inputs=inputs)
        self.ncols = ncols

    def read(self, index, force_type=np):
        grid = [[]]
        maxdim = np.zeros(shape=(3,), dtype=int)
        for r in self.inputs:
            if len(grid[-1]) == self.ncols:
                grid.append([])
            img = r.read(index=index, force_type=force_type)
            grid[-1].append(img)
            maxdim = np.maximum(maxdim, img.shape)
        res = np.zeros(shape=(maxdim[0] * len(grid), maxdim[1] * len(grid[0]), maxdim[2]), dtype=grid[0][0].dtype)
        for col in range(len(grid)):
            for row in range(len(grid[col])):
                img = grid[col][row]
                res[col * maxdim[0]: col * maxdim[0] + img.shape[0],
                row * maxdim[1]: row * maxdim[1] + img.shape[1]] = img
        return res


class Crop(VideoSupplier):
    def __init__(self, reader, x=0, y=0, width=-1, height=-1):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.last = (np.nan, None)

    def read(self, index, force_type=np):
        last = self.last
        if last[0] == index:
            return VideoSupplier.convert(last[1], force_type)
        img = self.inputs[0].read(index=index, force_type=force_type)
        res = img[self.x: self.x + self.height, self.y: self.y + self.width]
        self.last = (index, res)
        return res


class Math(VideoSupplier):
    def __init__(self, reader, expression, library='numpy'):
        super().__init__(n_frames=reader[0].n_frames, inputs=reader)
        if library == 'numpy':
            self.xp = np
        elif library == 'cupy':
            import cupy as cp
            self.xp = cp
        elif library == 'jax':
            import jax
            self.xp = jax.numpy
        else:
            raise Exception('Library ' + library + ' not known')
        self.exp = compile(expression, '<string>', 'exec')

    @staticmethod
    def name():
        return "math"

    def read(self, index, force_type=np):
        args = {'i' + str(i): self.inputs[i].read(index=index, force_type=self.xp) for i in range(len(self.inputs))}
        args['np'] = np
        args['xp'] = self.xp
        ldict = {}
        exec(self.exp, args, ldict)
        return VideoSupplier.convert(ldict['out'], force_type)


class MaxIndex(VideoSupplier):
    def __init__(self, reader, count, radius):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.count = int(count)
        self.radius = int(radius)

    @staticmethod
    def get_maxpixels(img, count, radius):
        import cv2
        res = np.zeros(shape=(count, 2), dtype=int)
        for i in range(count):
            maxpix = np.argmax(img)
            maxpix = np.unravel_index(maxpix, img.shape[0:2])
            res[i] = maxpix
            cv2.circle(img, (maxpix[1], maxpix[0]), radius, 0, -1)
            # maxpix=np.asarray(maxpix)
            # lhs = np.maximum(maxpix+radius, 0)
            # rhs = np.minimum(maxpix-radius, img.shape)
            # img[lhs[0]:rhs[0],lhs[1]:rhs[1]]=0
        return res

    @staticmethod
    def name():
        return "max"

    def read(self, index, force_type=None):
        img = self.inputs[0].read(index=index)
        img = VideoSupplier.convert(img, np)
        locations = MaxIndex.get_maxpixels(img, self.count, self.radius)
        values = img[(*locations.T,)]
        res = {}
        for i in range(self.count):
            cur = locations[i]
            res['x' + str(i)] = cur[0]
            res['y' + str(i)] = cur[1]
            res['c' + str(i)] = values[i]
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
        return np.asarray([int(x) for x in f], dtype=int)


def read_map(filename, source='from', destination='to', sourceoffset=0, destinationoffset=0):
    res = {}
    import pandas as pd
    csv = pd.read_csv(filename, sep=' ')

    def get_variable(csv, index):
        if isinstance(index, str):
            if index.isnumeric():
                index = int(index)
            elif len(index) != 0 and index[0] == '-' and index[1:].isnumeric():
                index = -int(index[1:])
        if isinstance(index, int):
            if index == -1:
                return np.arange(csv.shape[0])
            return np.asarray(csv.iloc[:, index])
        if isinstance(index, str):
            return np.asarray(csv[index])

    return dict(zip(get_variable(csv, source) + sourceoffset, get_variable(csv, destination) + destinationoffset))


class TimeToFrame(VideoSupplier):
    def __init__(self, reader, timingfile):
        import pandas as pd
        timings = pd.read_csv(timingfile)


class PermutateFrames(VideoSupplier):
    def __init__(self, reader, permutation=None, mapping=None, source='from', destination='to', sourceoffset=0,
                 destinationoffset=0, invalid_action="black"):
        if isinstance(permutation, str):
            permutation = read_numbers(permutation) + destinationoffset
        elif isinstance(mapping, str):
            permutation = read_map(mapping, source, destination, sourceoffset, destinationoffset)
        else:
            permutation = np.arange(destinationoffset, len(reader)) - sourceoffset
        self.permutation = permutation

        match (invalid_action):
            case "black":
                def invalid_black(index):
                    return self.invalid
                self.invalid_action = invalid_black
            case "exception":
                def invalid_exception(index):
                    return Exception(f"{index} not in range")
                self.invalid_action = invalid_exception
            case _:
                raise Exception(f"Action {invalid_action} not known")

        self.invalid = np.zeros_like(reader.read(index=0))
        if isinstance(self.permutation, dict):
            for frame in sorted(self.permutation.keys()):
                if self.permutation[frame] >= len(reader):
                    break
                n_frames = frame + 1
        else:
            n_frames = len(self.permutation)
        super().__init__(n_frames=n_frames, inputs=(reader,))

    def read(self, index, force_type=np):
        if index in self.permutation if isinstance(self.permutation, dict) else 0 <= index < len(self.permutation):
            return self.inputs[0].read(index=self.permutation[index], force_type=force_type)
        return self.invalid_action(index)


class BgrToGray(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames * 3, inputs=(reader,))

    def read(self, index, force_type=np):
        img = self.inputs[0].read(index=index // 3, force_type=force_type)
        return img[:, :, [index % 3]]


class ChangeFramerate(VideoSupplier):
    def __init__(self, reader, factor=1):
        super().__init__(n_frames=int(np.round(reader.n_frames / factor)), inputs=(reader,))
        self.factor = factor

    def read(self, index, force_type=np):
        return self.inputs[0].read(int(np.round(index * self.factor)), force_type=force_type)


class ConstFrame(VideoSupplier):
    def __init__(self, reader, frame):
        super().__init__(n_frames=reader.n_frames * 3, inputs=(reader,))
        self.frame = frame
        self.img = None

    def read(self, index, force_type=np):
        if self.img is None:
            self.img = self.inputs[0].read(self.frame, force_type=force_type)
        return VideoSupplier.convert(self.img, force_type)


class FrameDifference(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames - 1, inputs=(reader,))

    def read(self, index, force_type=np):
        return 128 + self.inputs[0].read(index=index + 1, force_type=force_type) - self.inputs[0].read(index=index,
                                                                                                       force_type=force_type)
