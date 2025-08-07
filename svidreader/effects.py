from svidreader.video_supplier import VideoSupplier
import numpy as np
import inspect


class Blur(VideoSupplier):
    def __init__(self, inputs, weights):
        super().__init__(n_frames=inputs[0].n_frames, inputs=inputs)
        self.weights = weights

    def read(self, index, force_type=np):
        result = None
        for idx, w in enumerate(self.weights):
            tmp = w * self.inputs.read(index + idx, force_type=force_type)
            if result is None:
                result = tmp
            else:
                result += tmp
        return result


class Image2Text(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))

    def read(self, index, force_type=np):
        import pytesseract
        img = self.inputs[0].read(index=index, force_type=force_type)
        return pytesseract.image_to_string(img, config='digits').strip()


class PixelCorrection(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.cache = {}

    def read(self, index, force_type=np):
        key_indices = np.copy(self.inputs[0].get_key_indices())
        nth_keyframe = np.searchsorted(key_indices, index, side="right")
        xp = force_type
        res = None
        firstframe = key_indices[nth_keyframe-1]
        correction = self.cache.get(firstframe, None)
        if correction is None:
            pixel_brightness = None
            lastframe = key_indices[nth_keyframe] if nth_keyframe < len(key_indices) else len(self.inputs[0])
            for i in range(firstframe, lastframe):
                image = self.inputs[0].read(index=i, force_type=force_type)
                image = image.astype(xp.uint16)
                if i == index:
                    res = image
                neighbours = np.median((xp.roll(image,-1,axis=0),
                                        xp.roll(image,1,axis=0),
                                        xp.roll(image,-1,axis=1),
                                        xp.roll(image,1,axis=1)), axis=0)
                if pixel_brightness is None:
                    pixel_brightness = image - neighbours
                else:
                    pixel_brightness += image - neighbours
            correction = pixel_brightness // (lastframe - firstframe)
            self.cache = {firstframe: correction}
        else:
            res = self.inputs[0].read(index, force_type=force_type)
        res = res - correction
        return xp.clip(res, 0, 255).astype(xp.uint8)


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


class MarkBorder(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))

    def read(self, index, force_type=np):
        image = self.inputs[0].read(index, force_type=force_type)
        if force_type is None:
            force_type = np
        xp = force_type
        border = xp.zeros_like(image, dtype=bool)
        masked = image > 128
        for i in range(2):
            for direction in (-1, 1):
                xp.logical_or(xp.roll(masked, direction, axis=i), border, out=border)
        xp.logical_and(border, ~masked, out=border)
        return border.astype(np.uint8) * 255


class ConvertColorspace(VideoSupplier):
    def __init__(self, reader, source, destination):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        if source == destination:
            self.functional = lambda x, force_type: x
        elif source == "rgb" and destination == "hsv":
            self.functional = lambda x, force_type: ConvertColorspace.rgb2hsv(x, force_type)
        elif source == "hsv" and destination == "rgb":
            self.functional = lambda x, force_type: ConvertColorspace.hsv2rgb(x, force_type)
        else:
            raise Exception(f"Conversion from {source} to {destination} not implemented")

    def read(self, index, force_type=np):
        return self.functional(self.inputs[0].read(index, force_type=force_type), force_type=force_type)

    @staticmethod
    def rgb2hsv(rgb, xp):
        """ convert RGB to HSV color space

        :param rgb: np.ndarray
        :return: np.ndarray
        """

        rgb = rgb.astype(xp.float32)
        maxv = xp.amax(rgb, axis=2)
        maxc = xp.argmax(rgb, axis=2)
        minv = xp.amin(rgb, axis=2)
        minc = xp.argmin(rgb, axis=2)

        hsv = xp.zeros(rgb.shape, dtype=xp.uint8)
        hsv[maxc == minc, 0] = xp.zeros(hsv[maxc == minc, 0].shape)
        hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + xp.spacing(1))) % 360.0)[
            maxc == 0]
        hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + xp.spacing(1))) + 120.0)[
            maxc == 1]
        hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + xp.spacing(1))) + 240.0)[
            maxc == 2]
        hsv[maxv == 0, 1] = xp.zeros(hsv[maxv == 0, 1].shape)
        hsv[maxv != 0, 1] = (1 - minv / (maxv + xp.spacing(1)))[maxv != 0]
        hsv[..., 2] = maxv
        return hsv

    @staticmethod
    def hsv2rgb(hsv, xp):
        """ convert HSV to RGB color space

        :param hsv: np.ndarray
        :return: np.ndarray
        """

        hi = xp.floor(hsv[..., 0] / 60.0) % 6
        hi = hi.astype(xp.uint8)
        v = hsv[..., 2].astype(xp.float32)
        f = (hsv[..., 0] / 60.0) - xp.floor(hsv[..., 0] / 60.0)
        p = v * (1.0 - hsv[..., 1])
        q = v * (1.0 - (f * hsv[..., 1]))
        t = v * (1.0 - ((1.0 - f) * hsv[..., 1]))

        rgb = xp.zeros(hsv.shape, dtype=xp.uint8)
        rgb[hi == 0, :] = xp.dstack((v, t, p))[hi == 0, :]
        rgb[hi == 1, :] = xp.dstack((q, v, p))[hi == 1, :]
        rgb[hi == 2, :] = xp.dstack((p, v, t))[hi == 2, :]
        rgb[hi == 3, :] = xp.dstack((p, q, v))[hi == 3, :]
        rgb[hi == 4, :] = xp.dstack((t, p, v))[hi == 4, :]
        rgb[hi == 5, :] = xp.dstack((v, p, q))[hi == 5, :]

        return rgb



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
        res = img[self.y: self.y + self.height, self.x: self.x + self.width]
        self.last = (index, res)
        return res


def video_generator(num_frames):
    return lambda functional: Functional([], functional, num_frames=num_frames)


def video_functional(functional):
    return lambda x, y=None: Functional([x] if y is None else [x, y], functional)


class Functional(VideoSupplier):
    def __init__(self, reader, functional, num_frames=None):
        if num_frames is None and len(reader) > 0:
            num_frames = reader[0].n_frames
        super().__init__(n_frames=num_frames, inputs=reader)
        self.functional = functional
        arguments = inspect.getfullargspec(functional)
        self.add_index = 'index' in arguments.args
        self.add_force_type = 'force_type' in arguments.args

    def read(self, index, force_type=np, **args):
        if self.add_index:
            args['index'] = index
        if self.add_force_type:
            args['force_type'] = force_type
        return self.functional(*[inp.read(index=index, force_type=force_type) for inp in self.inputs], **args)


def to_array(reader:VideoSupplier, jobs=1, show_progress=False, iterator=None, return_result=True, reduce=None, init=None):
    from svidreader import frame_iterator
    return frame_iterator.FrameIterator(input=reader, jobs=jobs, iterator=iterator).run(return_result=return_result, show_progress=show_progress, reduce=reduce, init=init)


def from_array(data):
    return ArrayReader(data)


class ArrayReader(VideoSupplier):
    def __init__(self, data):
        super().__init__(len(data), ())
        self.data = data

    def read(self, index, force_type=np):
        return VideoSupplier.convert(self.data[index], force_type)


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
            res[f'x{i}'] = cur[0]
            res[f'y{i}'] = cur[1]
            res[f'c{i}'] = values[i]
        return res


class Plot(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))

    def read(self, index, force_type=np):
        import cv2
        img = self.inputs[0].read(index=index, force_type=force_type)
        data = self.inputs[1].read(index=index, force_type=force_type)
        img = np.copy(img)
        cv2.circle(img, (data['x'], data['y']), 2, (255, 0, 0), data['c'])
        return img


class Scale(VideoSupplier):
    def __init__(self, reader, scale):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.scale = scale

    def read(self, index, force_type=np):
        lib = "cv"
        img = self.inputs[0].read(index=index, force_type=force_type)
        output_shape = (int(img.shape[0] * self.scale), int(img.shape[1] * self.scale))
        if lib == "skimage":
            from skimage.transform import resize
            resized = resize(img, output_shape)
        elif lib == "cv":
            import cv2
            resized = cv2.resize(img, output_shape[::-1])
        else:
            raise Exception(f"Library {lib} not known")
        return resized


def read_numbers(filename):
    with open(filename, 'r') as f:
        return np.asarray([int(x) for x in f], dtype=int)


def read_map(filename, source='from', destination='to', sourceoffset=0, destinationoffset=0):
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


class PermutateFrames(VideoSupplier):
    def __init__(self, reader, permutation=None, mapping=None, source='from', destination='to', sourceoffset=0,
                 destinationoffset=0, invalid_action="black"):
        if isinstance(permutation, str):
            permutation = read_numbers(permutation) + destinationoffset
        elif isinstance(mapping, str):
            permutation = read_map(mapping, source, destination, sourceoffset, destinationoffset)
        elif permutation is None:
            permutation = np.arange(destinationoffset, len(reader)) - sourceoffset
        self.permutation = permutation

        match invalid_action:
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
            n_frames = 0
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


class PrintEffect(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))

    def read(self, index, force_type=np):
        frame = self.inputs[0].read(index=index, force_type=force_type)
        print(frame)
        return frame

class TriggerEffect(VideoSupplier):
    def __init__(self,  passthrough_reader, trigger_reader):
        super().__init__(n_frames=passthrough_reader.n_frames, inputs=(passthrough_reader,*trigger_reader))

    def read(self, index, force_type=np):
        result = self.inputs[0].read(index=index, force_type=force_type)
        for i in range(1, len(self.inputs)):
            self.inputs[i].read(index=index, force_type=force_type)
        return result


class ConstFrame(VideoSupplier):
    def __init__(self, reader=None, frame=0, n_frames=-1, img=None):
        inputs = () if reader is None else (reader,)
        super().__init__(n_frames=n_frames, inputs=inputs)
        self.frame = frame
        self.img = img

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


class Reduction(VideoSupplier):
    def __init__(self, reader, operation="add"):
        from threading import Lock
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.result = None
        self.operation = operation
        self.num_frames = 0
        self.mutex = Lock()

    def read(self, index, force_type=np):
        img = self.inputs[0].read(index=index, force_type=force_type)
        with self.mutex:
            if self.result is None:
                self.result = img.astype(np.float32)
            else:
                match self.operation:
                    case "add": self.result += img
            self.num_frames += 1
        return self.result

    def get_result(self):
        return self.result / self.num_frames


class Overlay(VideoSupplier):
    def __init__(self, reader, overlay, x=0, y=0):
        super().__init__(n_frames=reader.n_frames, inputs=(reader, overlay))
        self.x, self.y = x, y
        self.overlay_index = (lambda index: 1) if overlay.n_frames == 1 else (lambda index: index)

    def read(self, index, force_type=np):
        img = self.inputs[0].read(index=index, force_type=force_type)
        overlay = self.inputs[1].read(index=self.overlay_index(index),
                                      force_type=force_type)

        coordinates = [self.x, self.y]
        for var, val in enumerate(coordinates):
            if isinstance(val, str):
                if val.isnumeric():
                    coordinates[var] = int(val)
                else:
                    variables = {'main_w': img.shape[1],
                                 'main_h': img.shape[0],
                                 'overlay_w': overlay.shape[1],
                                 'overlay_h': overlay.shape[0]}
                    coordinates[var] = int(eval(val, variables))
        x, y = coordinates

        if overlay.shape[2] == 4:
            dim = img.shape[2]
            alpha = overlay[:, :, 3]
            img[y:y + overlay.shape[0], x:x + overlay.shape[1]] = \
                ((img[y:y + overlay.shape[0], x:x + overlay.shape[1]] * (255 - alpha)
                 + overlay[:, :, 0:dim] * alpha)) // 255
        elif img.shape[2] == overlay.shape[2]:
            img[y:y + overlay.shape[0], x:x + overlay.shape[1]] = overlay
        else:
            overlay = overlay < 128
            overlay = np.repeat(overlay, img.shape[2], axis=2)
            img[y:y + overlay.shape[0], x:x + overlay.shape[1]][overlay] = 0
        return img
