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
        self.gauss = xp.fuse(MajorityVote.get_gauss(self.scale))
        self.foreground = foreground
        print(scale, window, foreground)

    @staticmethod
    def get_gauss(scale):
        scale = 1 / scale
        def gauss(x, y):
            diff = (x - y) * scale
            return xp.exp(-xp.sum(xp.square(diff), axis=2))
        return gauss

    def read(self, index, force_type=np):
        begin = max(0, index - self.window)
        end = min(index + self.window, self.n_frames)
        for i in range(begin, end):
            if i not in self.stack:
                self.stack[i] = self.inputs[0].read(index = i, force_type=xp)
        shape = self.stack[begin].shape[0:2]
        cache_next = {}
        for i in range(begin, end):
            curimage = self.stack[i].astype(xp.float32)
            if i in self.cache:
                ca  = self.cache[i]
                sum = xp.array(ca[2], dtype=xp.float32, copy=True)
                does_include = np.arange(ca[0], ca[1])
                should_include = np.arange(begin, end)
                for j in np.setdiff1d(should_include, does_include):
                    if j not in self.stack:
                        self.stack[j] = self.inputs[0].read(index=j, force_type=xp)
                    sum += self.gauss(curimage, self.stack[j])
                for j in np.setdiff1d(does_include, should_include):
                    if j not in self.stack:
                        self.stack[j] = self.inputs[0].read(index=j, force_type=xp)
                    sum -= self.gauss(curimage, self.stack[j])
            else:
                sum = xp.zeros(shape=shape, dtype=xp.float32)
                for j in range(begin, end):
                    sum += self.gauss(curimage, self.stack[j])
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
                xp.copyto(result, self.stack[i], where=mask[:,:,np.newaxis])
        for k, v in list(self.stack.items()):
            if k < begin or k > end:
                del self.stack[k]
        self.cache = cache_next
        return VideoSupplier.convert(result, force_type)