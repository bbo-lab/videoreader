from svidreader.video_supplier import VideoSupplier

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import numpy as np


class FrameIterator(VideoSupplier):
    def __init__(self, input, iterator=None, jobs=1, force_type=None):
        super().__init__(n_frames=input.n_frames, inputs=(input,))
        self.jobs = jobs
        self.force_type = force_type
        self.iterator = range(self.n_frames) if iterator is None else iterator

    def read(self, index, force_type=np, noreturn=False):
        image = self.inputs[0].read(index=index, force_type=force_type)
        return None if noreturn else image

    def run(self):
        if self.jobs != 1:
            thread_map(lambda index: self.read(index=index, force_type=self.force_type, noreturn=True), self.iterator, max_workers=self.jobs,
                       chunksize=1)
        else:
            for frame_idx in tqdm(self.iterator):
                self.read(frame_idx, force_type=self.force_type)
