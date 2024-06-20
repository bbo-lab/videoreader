from svidreader.video_supplier import VideoSupplier

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import numpy as np


class FrameIterator(VideoSupplier):
    def __init__(self, inputs, iterator=None, jobs=1, force_type=None):
        super().__init__(n_frames=inputs[0].n_frames, inputs=inputs)
        self.iterator = iterator

        iterator = range(self.n_frames) if iterator is None else iterator

        if jobs != 1:
            thread_map(self.read, iterator, kwargs={'force_type', force_type}, max_workers=jobs, chunksize=1)
        else:
            for frame_idx in tqdm(iterator):
                self.read(frame_idx, force_type=force_type)

    def read(self, index, force_type=np):
        return self.inputs[0].read(force_type=force_type)