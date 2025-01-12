import queue
from threading import Lock
from concurrent.futures import Future
import sys
from enum import IntEnum
from multiprocessing.pool import ThreadPool
import numpy as np
import logging
from svidreader.video_supplier import VideoSupplier

logger = logging.getLogger(__name__)


#This class acts as a cache-proxy to access a the-imageio-reader.
#Options are tuned to read compressed videos.
#The cache works with a seperate thread and tries to preload frames as good as possible

class CachedFrame:
    def __init__(self, data, last_used, hash):
        self.data = data
        self.png = None
        self.last_used = last_used
        self.hash = hash

    def memsize(self):
        if isinstance(self.data, np.ndarray):
            return self.data.nbytes
        return sys.getsizeof(self.data)


class QueuedLoad():
    def __init__(self, task, priority=0, future=None):
        self.priority = priority
        self.task = task
        self.future = future

    def __eq__(self, other):
        return self.priority == other.priority

    def __ne__(self, other):
        return self.priority != other.priority

    def __lt__(self, other):
        return self.priority < other.priority

    def __le__(self, other):
        return self.priority <= other.priority

    def __gt__(self, other):
        return self.priority > other.priority

    def __ge__(self, other):
        return self.priority >= other.priority


class PriorityThreadPool(ThreadPool):
    def __init__(self, processes=1):
        super().__init__(processes=processes)
        self.loadingQueue = queue.PriorityQueue()

    def close(self):
        self.loadingQueue = None
        super().close()

    def submit(self, task, priority=0, future=Future()):
        self.loadingQueue.put(QueuedLoad(task, priority=priority, future=future))
        self.apply_async(self.worker)
        return future

    def worker(self):
        try:
            if self.loadingQueue is None:
                return
            elem = self.loadingQueue.get(block=True, timeout=1)
            try:
                res = elem.task()
                elem.future.set_result(res)
            except Exception as e:
                elem.future.set_exception(e)
        except Exception as e:
            logger.log(logging.ERROR, f"timeout {e}")


class FrameStatus(IntEnum):
    NOT_CACHED = 0
    LOADING_QUEUED_BG = 1
    LOADING_QUEUED_FG = 2
    LOADING = 3
    CACHED = 4


class ImageCache(VideoSupplier):
    def __init__(self, reader, keyframes=None, maxcount=100, processes=1, preload=20, connect_segments=None):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        if self.n_frames > 0:
            self.framestatus = np.full(shape=(self.n_frames,), dtype=np.uint8, fill_value=FrameStatus.NOT_CACHED)
        else:
            self.framestatus = {}
        self.verbose = True
        self.lock = Lock()
        self.cached = {}
        self.maxcount = maxcount
        self.maxmemsize = 5000000000
        self.curmemsize = 0
        self.th = None
        self.usage_counter = 0
        self.last_read = 0
        self.num_preload_fwd = preload
        self.num_preload_bwd = preload
        self.connect_segments = preload if connect_segments is None else connect_segments
        self.keyframes = keyframes if keyframes is not None else reader.get_key_indices()
        self.ptp = PriorityThreadPool(processes=processes)

    def close(self, recursive=False):
        if self.ptp is None:
            logger.log(logging.ERROR, "closed multiple times")
        else:
            self.ptp.close()
            self.ptp = None
        super().close(recursive=recursive)

    def add_to_cache(self, index, data, hash):
        res = self.cached.get(index)
        if res is not None:
            self.curmemsize -= res.memsize()
            res.data = data
            res.last_used = self.usage_counter
            res.hash = hash
        else:
            res = CachedFrame(data, self.usage_counter, hash)
            self.cached[index] = res
        self.framestatus[index] = FrameStatus.CACHED
        self.curmemsize += res.memsize()
        self.usage_counter += 1
        return res

    def clean(self):
        if len(self.cached) > self.maxcount or self.curmemsize > self.maxmemsize:
            last_used = np.zeros(len(self.cached), dtype=int)
            keys = np.zeros(len(self.cached), dtype=int)
            i = 0
            for k, v in self.cached.items():
                keys[i] = k
                last_used[i] = v.last_used
                i += 1
            try:
                partition = np.argpartition(last_used, self.maxcount)
                oldmemsize = self.curmemsize
                oldsize = len(last_used)
                for p in partition[:max(0, len(last_used) - self.maxcount * 3 // 4)]:
                    k = keys[p]
                    self.curmemsize -= self.cached[k].memsize()
                    self.framestatus[k] = FrameStatus.NOT_CACHED
                    del self.cached[k]
                #print("cleaned", oldsize - len(self.cached),"of", oldsize, "freed",(oldmemsize - self.curmemsize)//1024//1024,"MB of",oldmemsize//1024//1024,"MB")
            except Exception as e:
                logger.log(logging.ERROR, e)

    def read_impl(self, index, force_type=None):
        while self.framestatus[index] == FrameStatus.LOADING:
            pass
        with self.lock:
            res = self.cached.get(index)
            if res is not None:
                return res
            self.framestatus[index] = FrameStatus.LOADING
        #Connect segments to not jump through the video
        if index - self.last_read < self.connect_segments:
            begin = self.last_read + 1
            if self.keyframes is not None:
                last_keyframe = np.searchsorted(self.keyframes, index, side="right") - 1
                if last_keyframe > 0:
                    begin = max(self.keyframes[last_keyframe], begin)
            for i in range(begin, index):
                with self.lock:
                    self.framestatus[i] = FrameStatus.LOADING
                data = self.inputs[0].read(index=i, force_type=force_type)
                with self.lock:
                    self.add_to_cache(i, data, 1 * 7 + index)
            self.last_read = index - 1
        elif self.last_read + 1 != index:
            logger.log(logging.DEBUG, f'nonsequential read {self.last_read} to {index}')
        with self.lock:
            self.framestatus[index] = FrameStatus.LOADING
        data = self.inputs[0].read(index=index, force_type=force_type)
        self.last_read = index
        with self.lock:
            res = self.add_to_cache(index, data, hash=1 * 7 + index)
            self.clean()
            return res

    def load(self, index, lazy=False, force_type=None):
        if lazy:
            if self.framestatus[index] > FrameStatus.NOT_CACHED:
                return
            self.framestatus[index] = FrameStatus.LOADING_QUEUED_BG
            priority = index + self.num_preload_bwd
        else:
            if self.framestatus[index] != FrameStatus.LOADING:
                self.framestatus[index] = FrameStatus.LOADING_QUEUED_FG
            priority = index
        future = Future()
        return self.ptp.submit(lambda: self.read_impl(index, force_type=force_type), priority=priority, future=future)

    def get_result_from_future(self, future):
        res = future.result()
        res.last_used = self.usage_counter
        self.usage_counter += 1
        return res.data

    def read(self, index=None, blocking=True, force_type=np):
        if index >= self.n_frames:
            raise Exception('Out of bounds, frame ' + str(index) + ' of ' + str(self.n_frames) + 'requested')
        with self.lock:
            res = self.cached.get(index)
        if res is None:
            future = self.load(index, lazy=False, force_type=force_type)
        end = index
        if self.n_frames > 0:
            end = min(index + self.num_preload_fwd, self.n_frames - 1)
        begin = max(index - self.num_preload_bwd, 0)
        if self.keyframes is not None:
            right_idx = self.keyframes.searchsorted(index, 'right') - 1
            begin = 0 if right_idx == 0 else self.keyframes[right_idx]
            begin = max(begin, index - self.maxcount // 2)
        for i in range(begin, end):
            self.load(index=i, lazy=True, force_type=force_type)
        if res is not None:
            res.last_used = self.usage_counter
            self.usage_counter += 1
            return VideoSupplier.convert(res.data, force_type)
        if blocking:
            res = self.get_result_from_future(future)
            return VideoSupplier.convert(res, force_type)
        future.add_done_callback(lambda: self.get_result_from_future(future))
        return future
