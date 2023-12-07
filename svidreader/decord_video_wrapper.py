from decord import VideoReader, gpu
import numpy as np
from svidreader.video_supplier import VideoSupplier
from threading import Thread, Lock
import time
import logging


class DecordVideoReader(VideoSupplier):
    def __init__(self, filename):
        super().__init__(n_frames=0, inputs=())
        self.closed = False
        self.filename = filename
        self.video = filename
        self.vr = VideoReader(filename)
        self.n_frames = len(self.vr)
        self.count = 0
        self.t = Thread(target=self.seek_end)
        self.mutex = Lock()
        self.last_read = 0
        self.last_index = 0
        self.t.start()


    def seek_end(self):
        while(not self.closed):
            time.sleep(1)
            with self.mutex:
                if time.time() - self.last_read > 5:
                    self.vr.seek(self.n_frames - 1)
                    self.last_read = np.inf

    def get_key_indices(self):
        return np.asarray(self.vr.get_key_indices(),dtype=int)

    def close(self):
        self.closed = True
        super().close()

    def read(self, index, force_type=np):
        with self.mutex:
            frame = self.vr.get_batch([index]).asnumpy()
            self.last_read = time.time()
        if index != self.last_index + 1:
            logging.debug(f"Non-sequential {self.last_index} to {index}")
        self.count += 1
        self.last_index = index
        return VideoSupplier.convert(frame[0], force_type)


    def get_meta_data(self):
        return {"key_indices": self.vr.get_key_indices(), "fps": self.vr.get_avg_fps()}