import threading
import queue
from threading import RLock
import concurrent.futures
from concurrent.futures import Future
import copy
from svidreader.video_supplier import VideoSupplier



#This class acts as a cache-proxy to access a the-imageio-reader.
#Options are tuned to read compressed videos.
#The cache works with a seperate thread and tries to preload frames as good as possible

class CachedFrame:
    def __init__(self, data, last_used, hash):
        self.data = data
        self.last_used = last_used
        self.hash = hash


class QueuedLoad():
    def __init__(self, task, priority = 0, future = None):
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


class PriorityThreadPool:
    def __init__(self):
        self.loadingQueue = queue.PriorityQueue()
        self.exit = threading.Event()
        self.wakeup = threading.Event()
        self.th = threading.Thread(target=self.worker, daemon=True)
        self.th.start()


    def close(self):
        self.exit.set()
        self.wakeup.set()
        

    def submit(self,task, priority=0):
        future = Future()
        self.loadingQueue.put(QueuedLoad(task, priority = priority, future = future))
        self.wakeup.set()
        return future


    def worker(self):
        while not self.exit.is_set():
            self.wakeup.wait()
            if self.wakeup.is_set():
                self.wakeup.clear()
                while not self.loadingQueue.empty():
                    elem = self.loadingQueue.get()
                    res = elem.task()
                    elem.future.set_result(res)


class ImageCache(VideoSupplier):
    def __init__(self, reader, n_frames = 0, keyframes = None):
        super().__init__(n_frames=n_frames, inputs=(reader,))
        self.rlock = RLock()
        self.cached = {}
        self.maxsize = 100
        self.th = None
        self.usage_counter = 0
        self.last_read = 0
        self.num_preload = 20
        self.connect_segments = 20
        self.n_frames = n_frames
        self.keyframes = keyframes
        self.ptp = PriorityThreadPool()


    def close(self):
        self.ptp.close()
        super().close()


    def add_to_cache(self, index, data, hash):
        res = CachedFrame(data, self.usage_counter, hash)
        self.cached[index] = res
        self.usage_counter += 1
        return res


    def clean(self):
        if len(self.cached) > self.maxsize:
                for key in [k for k in self.cached]:
                    if self.cached[key].last_used < self.usage_counter - self.maxsize:
                        del self.cached[key]


    def read_impl(self,index):
        with self.rlock:
             res = self.cached.get(index)
             if res is not None:
                 return res
        #Connect segments to not jump through the video
        if index - self.last_read < self.connect_segments:
            for i in range(self.last_read + 1, index):
                data = self.inputs[0].read(index=i)
                with self.rlock:
                    self.add_to_cache(i, data, hash(self.inputs[0]) * 7 + index)
        data = self.inputs[0].read(index=index)
        last_read = index
        self.last_read = index
        with self.rlock:
            res = self.add_to_cache(index, data, hash(self.inputs[0]) * 7 + index)
            self.clean()
            return res


    def preload(self, index):
        with self.rlock:
             if index in self.cached:
                return
        self.ptp.submit(lambda: self.read_impl(index=index), priority = index)


    def get_result_from_future(self, future):
        res = future.result()
        res.last_used = self.usage_counter
        self.usage_counter += 1
        return res.data


    def read(self,index=None,blocking=True):
        with self.rlock:
            res = self.cached.get(index)
            if res is not None:
                for i in range(max(index - self.num_preload,0), min(index + self.num_preload,self.n_frames - 1), 1):
                    self.preload(index = i)
                res.last_used = self.usage_counter
                self.usage_counter += 1
                return res.data

        future = self.ptp.submit(lambda : self.read_impl(index), priority = index - 1000000)
        for i in range(max(index - self.num_preload,0), index + self.num_preload, 1):
            self.preload(i)
        if blocking:
            return self.get_result_from_future(future)
        future.add_done_callback(lambda : self.get_result_from_future(future))

    def __next__(self):
        if (self.frame_idx + 1) < self.n_frames:
            self.frame_idx += 1
            return self.read(self.frame_idx)
        else:
            print("Reached end")
            raise StopIteration
