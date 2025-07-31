from svidreader.video_supplier import VideoSupplier

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import numpy as np
from threading import Lock


class FrameIterator(VideoSupplier):
    def __init__(self, input:VideoSupplier, iterator=None, jobs=1, force_type=None):
        super().__init__(n_frames=input.n_frames, inputs=(input,))
        self.jobs = jobs
        self.force_type = force_type
        self.iterator = range(self.n_frames) if iterator is None else iterator

    def read(self, index, force_type=np, return_result=True):
        image = self.inputs[0].read(index=index, force_type=force_type)
        return image if return_result else None

    def run(self, return_result=False, show_progress = False, init=None, reduce=None):
        result = []
        aggregate = init
        if reduce is not None:
            return_result = False
        if reduce is None:
            def functional(frame_idx):
                return self.read(index=frame_idx, force_type=self.force_type, return_result=return_result)
        else:
            lock = Lock()
            def functional(frame_idx):
                nonlocal aggregate
                tmp = self.read(index=frame_idx, force_type=self.force_type, return_result=True)
                with lock:
                    aggregate = reduce(aggregate, tmp)

        if self.jobs != 1:
            args = {}
            if self.jobs != 0:
                args = {"max_workers": self.jobs}
            result = thread_map(functional, self.iterator, **args, chunksize=1, disable=not show_progress)
        else:
            for frame_idx in tqdm(self.iterator, disable=not show_progress):
                tmp = functional(frame_idx)
                if return_result:
                    result.append(tmp)
        if reduce is not None:
            return aggregate
        return result if return_result else None