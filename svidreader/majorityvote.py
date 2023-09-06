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
        self.foreground = foreground
        print(scale, window, foreground)

    @xp.fuse()
    def gauss(x, y, scale):
        diff = (x - y) * (1 / scale)
        return xp.exp(-xp.sum(xp.square(diff), axis=2))

    def read(self, index):
        begin = max(0, index - self.window)
        end = min(index + self.window, self.inputs[0].n_frames)
        for i in range(begin, end):
            if i not in self.stack:
                self.stack[i] = xp.asarray(self.inputs[0].read(index = i))
        best_sum = xp.full(fill_value=-np.inf, shape=self.stack[begin].shape[0:2], dtype=xp.float32)
        result  = xp.copy(self.stack[index])
        cache_next = {}
        for i in range(begin, end):
            sum = xp.zeros_like(best_sum)
            curimage = self.stack[i].astype(xp.float32)
            if i in self.cache :
                ca  = self.cache[i]
                sum += ca[2]
                does_include = np.arange(ca[0], ca[1])
                should_include = np.arange(begin, end)
                for j in np.setdiff1d(should_include, does_include):
                    if j not in self.stack:
                        self.stack[j] = xp.asarray(self.inputs[0].read(index=j))
                    sum += self.gauss( curimage, self.stack[j], self.scale)
                for j in np.setdiff1d(does_include, should_include):
                    if j not in self.stack:
                        self.stack[j] = xp.asarray(self.inputs[0].read(index=j))
                    sum -= self.gauss(curimage, self.stack[j], self.scale)
            else:
                for j in range(begin, end):
                    sum += self.gauss(curimage, self.stack[j], self.scale)
            cache_next[i] = (begin, end, sum)
            if self.foreground:
                sum = -sum
            mask = best_sum < sum
            best_sum[mask] = sum[mask]
            result[mask] = self.stack[i][mask]
        for k, v in list(self.stack.items()):
            if k < begin or k > end:
                del self.stack[k]
        self.cache = cache_next
        return xp.asnumpy(result)