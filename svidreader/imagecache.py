import threading
import queue
from threading import RLock
from datetime import datetime
import concurrent.futures
from concurrent.futures import Future
from time import sleep
import copy


#This class acts as a cache-proxy to access a the-imageio-reader.
#Options are tuned to read compressed videos.
#The cache works with a seperate thread and tries to preload frames as good as possible

class CachedFrame:
    def __init__(self, data, last_used):
        self.data = data
        self.last_used = last_used


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
        self.th = threading.Thread(target=self.worker, daemon=True)
        self.th.start()


    def close(self):
        self.exit.set()
        

    def submit(self,task, priority=0):
        future = Future()
        self.loadingQueue.put(QueuedLoad(task, priority = priority, future = future))
        return future


    def worker(self):
        while not self.exit.is_set():
            sleep(0.001)
            while not self.loadingQueue.empty():
                elem = self.loadingQueue.get()
                res = elem.task()
                elem.future.set_result(res)


class ImageCache:
    def __init__(self, reader, n_frames = 0):
        self.reader = reader
        self.rlock = RLock()
        self.cached = {}
        self.maxsize = 100
        self.th = None
        self.usage_counter = 0
        self.last_read = 0
        self.num_preload = 20
        self.connect_segments = 20
        self.n_frames = n_frames
        self.ptp = PriorityThreadPool()


    def close(self):
        self.reader.close()
        self.ptp.close()


    def add_to_cache(self, index, data):
        res = CachedFrame(data, self.usage_counter)
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
                data = self.reader.read(index=i)
                with self.rlock:
                    self.add_to_cache(i, data)
        data = self.reader.read(index=index)
        last_read = index
        self.last_read = index
        with self.rlock:
            res = self.add_to_cache(index, data)
            self.clean()
            return res


    def preload(self, index):
        self.ptp.submit(lambda: self.read_impl(index=index), priority = index)


    def read(self,index=None):
        with self.rlock:
            res = self.cached.get(index)
            if res is not None:
                for i in range(max(index - self.num_preload,0), min(index + self.num_preload,self.n_frames), 1):
                    self.preload(index = i)
                res.last_used = self.usage_counter
                self.usage_counter += 1
                return res.data

        future = self.ptp.submit(lambda : self.read_impl(index), priority = index - 1000000)
        for i in range(max(index - self.num_preload,0), index + self.num_preload, 1):
            self.preload(i)
        res = future.result()
        res.last_used = self.usage_counter
        self.usage_counter += 1
        return res.data
