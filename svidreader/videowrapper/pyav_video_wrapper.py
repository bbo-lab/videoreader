from svidreader.video_supplier import VideoSupplier
import av
import numpy as np


class PyAvVideoReader(VideoSupplier):
    def __init__(self, filename):
        container = av.open(filename)

        self.container = container
        self.stream = container.streams.video[0]

        super().__init__(
            n_frames=self.stream.frames,
            inputs=()
        )

        self.fps = float(self.stream.average_rate)

    def get_fps(self):
        return self.fps

    def get_key_indices(self):
        return np.arange(self.stream.frames)

    def read(self, index, force_type=np):

        # approximate timestamp
        timestamp = int(
            (index / self.fps) / self.stream.time_base
        )

        self.container.seek(
            timestamp,
            stream=self.stream,
            any_frame=False,
            backward=True
        )

        current_index = None

        for frame in self.container.decode(self.stream):

            if current_index is None:
                # estimate first decoded frame index
                current_index = int(
                    frame.pts * self.stream.time_base * self.fps
                )

            if current_index >= index:
                arr = frame.to_ndarray(format="rgb24")
                return VideoSupplier.convert(arr, force_type)

            current_index += 1

        raise IndexError(f"Frame {index} not found")

    def get_meta_data(self):
        return {
            "fps": self.get_fps()
        }

    def close(self, recursive=False):
        self.container.close()