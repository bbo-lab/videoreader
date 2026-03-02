from svidreader.video_supplier import VideoSupplier
import numpy as np
try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp

class MajorityVote(VideoSupplier):
    def __init__(self, reader, window, scale, foreground = False):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.window = window
        self.scale = float(scale)
        self.cache = {}
        self.stack = {}
        self.distance_function = MajorityVote.get_gauss(self.scale, xp=xp)
        self.foreground = foreground

    @staticmethod
    def get_gauss(scale, xp=np):
        scale = 1 / scale
        def gauss(x, y):
            diff = (x - y) * scale
            return xp.exp(-xp.sum(xp.square(diff), axis=2))
        if xp == np:
            return gauss
        else:
            return xp.fuse(gauss)

    def add_to_stack(self, index:int) -> xp.ndarray:
        if index not in self.stack:
            self.stack[index] = self.inputs[0].read(index = index, force_type=xp)
        return self.stack[index]

    def read(self, index, force_type=np):
        begin = max(0, index - self.window)
        end = min(index + self.window, self.n_frames)
        for i in range(begin, end):
            self.add_to_stack(index=i)
        shape = self.stack[begin].shape[0:2]
        cache_next = {}
        should_include = np.arange(begin, end)
        for i in range(begin, end):
            curimage = self.stack[i].astype(xp.float32, copy=False)
            if i in self.cache:
                ca  = self.cache[i]
                sum = xp.array(ca[2], dtype=xp.float32, copy=True)
                does_include = np.arange(ca[0], ca[1])
                for j in np.setdiff1d(should_include, does_include):
                    sum += self.distance_function(curimage, self.add_to_stack(j))
                for j in np.setdiff1d(does_include, should_include):
                    sum -= self.distance_function(curimage, self.add_to_stack(j))
            else:
                sum = xp.zeros(shape=shape, dtype=xp.float32)
                for j in range(begin, end):
                    sum += self.distance_function(curimage, self.add_to_stack(j))
            cache_next[i] = (begin, end, sum)
            if i == begin:
                best_sum = xp.copy(sum)
                result = xp.copy(self.stack[i])
            else:
                if self.foreground:
                    mask = best_sum > sum
                else:
                    mask = best_sum < sum
                xp.copyto(best_sum, sum, where = mask)
                xp.copyto(result, self.stack[i], where=mask[:,:,xp.newaxis])
        for k, v in list(self.stack.items()):
            if k < begin or k > end:
                del self.stack[k]
        self.cache = cache_next
        return VideoSupplier.convert(result, force_type)