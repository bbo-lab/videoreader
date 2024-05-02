from svidreader.video_supplier import VideoSupplier
import numpy as np


class DumpToFile(VideoSupplier):
    def __init__(self, reader, outputfile, writer=None, opts={}, makedir=False, comment=None):
        import imageio
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.outputfile = outputfile
        self.output = None
        self.pipe = None
        self.opts = opts
        if makedir:
            from pathlib import Path
            Path(outputfile).parent.mkdir(parents=True, exist_ok=True)
        if writer is not None and writer == "ffmpeg":
            self.type = "ffmpeg_movie"
        elif outputfile.endswith('.mp4'):
            self.type = "movie"
            self.outputfile = outputfile
        elif outputfile.endswith('.zip'):
            self.type = "zip"
        else:
            self.type = "csv"
            self.mapkeys = None
            self.output = open(outputfile, 'w')
            if comment is not None:
                self.output.write(comment + '\n')

    def close(self):
        super().close()
        if self.output is not None:
            self.output.close()
        if self.pipe is not None:
            self.pipe.stdin.close()

    def read(self, index, force_type=np):
        data = self.inputs[0].read(index=index, force_type=force_type)
        if self.type == "movie":
            import imageio
            if self.output is None:
                self.output = imageio.get_writer(self.outputfile, fps=200, quality=8)
            if data is not None:
                self.output.append_data(data)
        elif self.type == "csv":
            if self.mapkeys == None and isinstance(data, dict):
                self.mapkeys = data.keys()
                self.output.write(f"index {' '.join(self.mapkeys)} \n")
            self.output.write(f"{index} {' '.join([str(data[k]) for k in self.mapkeys])} \n")
        elif self.type == "zip":
            import cv2
            import zipfile
            if self.output is None:
                self.output = zipfile.ZipFile(self.outputfile, mode="w")
            img_name = "{:06d}.png".format(index)
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
            if index % 10 != 0:
                data = np.copy(data)
                data -= self.inputs[0].read(index=(index // 10) * 10)
                data += 127
            png_encoded = cv2.imencode('.png', data, encode_param)[1].tostring()
            self.output.writestr(img_name, png_encoded)
        elif self.type == "ffmpeg_movie":
            import subprocess as sp
            import os
            if self.pipe is None:
                encoder = self.opts.get('encoder','libx264')
                if encoder is None:
                    encoder = 'hevc_nvenc'
                if encoder == 'hevc_nvenc':
                    codec = ['-i', '-', '-an', '-vcodec', 'hevc_nvenc']
                elif encoder == 'h264_nvenc':
                    codec = ['-i', '-', '-an', '-vcodec', 'h264_nvenc']
                elif encoder == '264_vaapi':
                    codec = ['-hwaccel', 'vaapi' '-hwaccel_output_format', 'hevc_vaapi', '-vaapi_device',
                             '/dev/dri/renderD128', '-i',
                             '-', '-an', '-c:v', 'hevc_vaapi']
                elif encoder == 'uncompressed':
                    codec = ['-f', 'rawvideo']
                elif encoder == 'libx264':
                    codec = ['-i', '-', '-vcodec', 'libx264']
                elif encoder == 'h264_v4l2m2m':
                    codec = ['-i', '-', '-c:v', 'h264_v4l2m2m']
                elif encoder == 'dummy':
                    codec = ['null']
                else:
                    raise Exception("Encoder " + args.encoder + " not known")
                pix_fmt = 'rgb24'
                if data.shape[2] == 1:
                    pix_fmt = 'gray8'
                command = ["ffmpeg",
                           '-y',  # (optional) overwrite output file if it exists
                           '-f', 'rawvideo',
                           '-vcodec', 'rawvideo',
                           '-s', f'{data.shape[1]}x{data.shape[0]}',  # size of one frame
                           '-pix_fmt', pix_fmt,
                           '-r', '200',  # frames per second
                           '-rtbufsize', '2G',
                           *codec,
                           '-preset', self.opts.get('preset', 'slow'),
                           '-qmin', '10',
                           '-qmax', '26',
                           '-b:v', self.opts.get('bitrate', '10M'),
                           self.outputfile]
                self.pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.STDOUT, bufsize=1000, preexec_fn=os.setpgrp)
            self.pipe.stdin.write(data.tobytes())
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


class Concatenate(VideoSupplier):
    def __init__(self, inputs):
        super().__init__(n_frames=np.sum([inp.n_frames for inp in inputs]), inputs=inputs)
        self.videostarts = np.cumsum([0] + [inp.n_frames for inp in inputs])

    def read(self, index, force_type=np):
        iinput = np.searchsorted(self.videostarts, index, side='right') - 1
        index = index - self.videostarts[iinput]
        return self.inputs[iinput].read(index, force_type=force_type)


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
        img = np.copy(img)
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
        img = self.inputs[0].read(index=index, force_type=np)
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


class GrayToBgr(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames // 3, inputs=(reader,))

    def read(self, index, force_type=np):
        return np.dstack([self.inputs[0].read(index=index * 3 + i, force_type=force_type) for i in range(3)])


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


class Overlay(VideoSupplier):
    def __init__(self, reader, overlay, x=0, y=0):
        super().__init__(n_frames=reader.n_frames, inputs=(reader, overlay))
        self.x = x
        self.y = y
        for var in ['x', 'y']:
            if isinstance(getattr(self, var), str):
                val = getattr(self, var)
                if val.isnumeric():
                    setattr(self, var, int(val))
                else:
                    main_h, main_w, _ = reader.get_shape()
                    overlay_h, overlay_w, _ = overlay.get_shape()
                    locals_dict = locals()
                    for rep_key in ['main_w', 'main_h', 'overlay_w', 'overlay_h']:
                        val = val.replace(rep_key, str(locals_dict[rep_key]))
                    setattr(self, var, int(eval(val)))

        self.overlay_index = lambda index: index
        if reader.n_frames != overlay.n_frames:
            self.overlay_index = lambda index: 0

        self.overlay_mode = 'replace'
        if reader.get_shape()[2] != overlay.get_shape()[2]:
            self.overlay_mode = 'bool'

    def read(self, index, force_type=np):
        img = self.inputs[0].read(index=index, force_type=force_type)
        overlay = self.inputs[1].read(index=self.overlay_index(index),
                                      force_type=force_type)
        if self.overlay_mode == 'replace':
            img[self.y:self.y + overlay.shape[0], self.x:self.x + overlay.shape[1]] = overlay
        elif self.overlay_mode == 'bool':
            overlay = overlay < 128
            overlay = np.repeat(overlay, img.shape[2], axis=2)
            img[self.y:self.y + overlay.shape[0], self.x:self.x + overlay.shape[1]][overlay] = 0
        return img
