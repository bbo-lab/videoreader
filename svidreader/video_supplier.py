class VideoSupplier:
    def __init__(self, n_frames, inputs = ()):
        self.inputs = inputs
        self.n_frames = n_frames

    def __iter__(self):
        return self

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

    def improps(self):
        return self.inputs[0].improps()

    def get_meta_data(self):
        return self.inputs[0].get_meta_data()

    def get_data(self, index):
        return self.read(index)

    def __hash__(self):
        res = hash(self.__class__.__name__)
        for i in self.inputs:
            res = res * 7 + hash(i)
        return res

    def __next__(self):
        if (self.frame_idx + 1) < self.n_frames:
            self.frame_idx += 1
            return self.read(self.frame_idx)
        else:
            print("Reached end")
            raise StopIteration