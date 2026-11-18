from decord import VideoReader
import numpy as np
from svidreader.video_supplier import VideoSupplier
from threading import Thread, Lock
import time
import logging
import subprocess
import json


class DecordVideoReader(VideoSupplier):
    def __init__(self, filename):
        vr = VideoReader(filename)
        super().__init__(n_frames=len(vr), inputs=())

        self.closed = False
        self.filename = filename
        self.video = filename
        self.vr = vr
        self.count = 0
        self.t = Thread(target=self.seek_end)
        self.mutex = Lock()
        self.last_read = 0
        self.last_index = 0

        # Try detecting whether the source video is grayscale.
        self.is_grayscale = False
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=pix_fmt",
                    "-of", "json",
                    filename
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )

            streams = json.loads(result.stdout).get("streams", [])
            if streams:
                pix_fmt = streams[0].get("pix_fmt", "")
                self.is_grayscale = pix_fmt.startswith("gray")

            logging.debug(
                f"Video pixel format: {pix_fmt if streams else 'unknown'}, "
                f"grayscale: {self.is_grayscale}"
            )

        except Exception as e:
            logging.warning(
                f"Could not probe video pixel format for {filename}: {e}. "
                "Using default RGB behavior."
            )

        self.t.start()

    def get_fps(self):
        return self.vr.get_avg_fps()

    def seek_end(self):
        while not self.closed:
            time.sleep(1)
            with self.mutex:
                if time.time() - self.last_read > 5:
                    self.vr.seek(self.n_frames - 1)
                    self.last_read = np.inf

    def get_key_indices(self):
        return np.asarray(self.vr.get_key_indices(), dtype=int)

    def close(self, recursive=False):
        self.closed = True
        super().close(recursive=recursive)

    def read(self, index, force_type=None):
        is_batch = np.ndim(index) > 0
        indices = index if is_batch else [index]

        with self.mutex:
            frame = self.vr.get_batch(indices).asnumpy()
            self.last_read = time.time()

        if self.is_grayscale:
            frame = frame[..., :1]

        if not is_batch:
            frame = frame[0]

        if indices[0] != self.last_index + 1:
            logging.debug(f"Non-sequential {self.last_index} to {indices[0]}")

        self.count += len(indices)
        self.last_index = indices[-1]

        return VideoSupplier.convert(frame, force_type)

    def get_meta_data(self):
        return {
            "key_indices": self.vr.get_key_indices(),
            "fps": self.vr.get_avg_fps()
        }
