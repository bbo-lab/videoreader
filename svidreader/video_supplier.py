import inspect
import logging

logger = logging.getLogger(__name__)

class VideoSupplier:
    def __init__(self, n_frames, inputs = ()):
        self.inputs = inputs
        self.n_frames = n_frames
        self.shape = None

    def __iter__(self):
        return VideoIterator(reader=self)

    def __len__(self):
        return self.n_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        for input in self.inputs:
            input.close()

    def get_key_indices(self):
        return self.inputs[0].get_key_indices()

    def get_shape(self):
        if self.shape is None:
            self.shape = self.read(0).shape
        return self.shape

    def get_offset(self):
        if len(self.inputs[0]) == 0:
            return (0,0)
        return self.inputs[0].get_offset()

    def get_meta_data(self):
        if len(self.inputs) == 0:
            return {}
        return self.inputs[0].get_meta_data()

    def read(self):
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
        if module == None:
            return img
        t = type(img)
        if inspect.getmodule(t) == module:
            return img
        if logging.DEBUG >= logging.root.level:
            finfo = inspect.getouterframes(inspect.currentframe())[1]
            logger.log(logging.DEBUG, F'convert {t.__module__} to {module.__name__} by {finfo.filename} line {finfo.lineno}')
        if t.__module__ == 'cupy':
            return module.array(img.get(), copy=False)
        return module.array(img,copy=False)

class VideoIterator(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames = reader.n_frames, inputs=(reader,))
        self.frame_idx = 0

    def __next__(self):
        if self.frame_idx < self.n_frames:
            res = self.inputs[0].read(self.frame_idx)
            self.frame_idx += 1
            return res
        else:
            raise StopIteration