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
        return self.inputs[0].get_meta_data()

    def get_data(self, index):
        return self.read(index)

    def __hash__(self):
        res = hash(self.__class__.__name__)
        for i in self.inputs:
            res = res * 7 + hash(i)
        return res

class VideoIterator(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames = reader.n_frames, inputs=(reader,))
        self.frame_idx = 0

    def __next__(self):
        if self.frame_idx + 1 < self.n_frames:
            self.frame_idx += 1
            return self.inputs[0].read(self.frame_idx)
        else:
            raise StopIteration