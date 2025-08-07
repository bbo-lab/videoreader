import inspect
import logging
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class ARRAY_MODULE_BEHAVIOR(Enum):
    DEFAULT = 0
    NATIVE = 1


class VideoSupplier:
    def __init__(self, n_frames, inputs=()):
        self.inputs = inputs
        self.n_frames = n_frames
        self.shape = None
        self.default_array_module = np
        self.is_alive = True
        self.num_entered = 0

    def __iter__(self):
        return VideoIterator(reader=self)

    def __len__(self):
        return self.n_frames

    def __enter__(self):
        if self.num_entered == 0:
            for inp in self.inputs:
                inp.__enter__()
        self.num_entered += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.num_entered -= 1
        if self.num_entered == 0:
            if self.inputs is not None:
                for inp in self.inputs:
                    inp.__exit__(exc_type, exc_value, traceback)
            self.close()

    def __del__(self):
        self.close()

    def set_default_array_module(self, default_array_module):
        self.default_array_module = default_array_module

    def __getitem__(self, key):
        return self.read(key, self.default_array_module)

    def close(self, recursive=False):
        if self.is_alive:
            self.is_alive = False
            if recursive:
                for input in self.inputs:
                    input.close(recursive=recursive)
        self.inputs = None

    def get_key_indices(self):
        return self.inputs[0].get_key_indices()

    def get_shape(self):
        if self.shape is None:
            self.shape = self.read(0).shape
        return self.shape

    def get_offset(self):
        if len(self.inputs[0]) == 0:
            return 0, 0
        return self.inputs[0].get_offset()

    def get_meta_data(self):
        if len(self.inputs) == 0:
            return {}
        return self.inputs[0].get_meta_data()

    def read(self, index, force_type=np):
        raise NotImplementedError("This method has to be overriden")

    def get_data(self, index):
        return self.read(index)

    def __hash__(self):
        res = hash(self.__class__.__name__)
        for i in self.inputs:
            res = res * 7 + hash(i)
        return res

    @staticmethod
    def convert(img, module):
        if isinstance(img, str):
            return img
        if module == None:
            return img
        t = type(img)
        if inspect.getmodule(t) == module:
            return img
        if logging.DEBUG >= logging.root.level:
            finfo = inspect.getouterframes(inspect.currentframe())[1]
            logger.log(logging.DEBUG,
                       F'convert {t.__module__} to {module.__name__} by {finfo.filename} line {finfo.lineno}')
        if t.__module__ == 'cupy':
            return module.array(img.get(), copy=False)
        return module.array(img, copy=False)


    @staticmethod
    def get_array_module(name):
        match name:
            case "cupy":
                import cupy as xp
            case "numpy":
                import numpy as xp
            case _:
                raise Exception(f"{name} must be one of (cupy, numpy)")
        return xp


class VideoIterator(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.frame_idx = 0

    def __next__(self):
        if self.frame_idx < self.n_frames:
            res = self.inputs[0].read(self.frame_idx)
            self.frame_idx += 1
            return res
        else:
            raise StopIteration
