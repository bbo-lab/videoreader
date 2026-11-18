from svidreader.video_supplier import VideoSupplier
import av
import numpy as np


class PyAvVideoReader(VideoSupplier):
    def __init__(
            self,
            filename,
            hwaccel="vaapi",
            hwaccel_device=None,
    ):
        # Configure hardware decoding before opening the container.
        if hwaccel is not None:
            hwaccel_kwargs = {
                "device_type": hwaccel,
                "allow_software_fallback": False,
            }

            if hwaccel_device is not None:
                hwaccel_kwargs["device"] = hwaccel_device

            hwaccel_ctx = av.codec.hwaccel.HWAccel(**hwaccel_kwargs)

            container = av.open(
                filename,
                hwaccel=hwaccel_ctx,
            )
        else:
            container = av.open(filename)

        self.container = container
        self.stream = container.streams.video[0]

        n_frames = self.stream.frames
        fps = float(self.stream.average_rate)

        if not n_frames:
            if self.stream.duration is not None:
                duration_sec = (
                        float(self.stream.duration)
                        * float(self.stream.time_base)
                )

            elif container.duration is not None:
                duration_sec = (float(container.duration) / av.time_base)

            else:
                duration_sec = 0

            n_frames = round(duration_sec * fps)

        super().__init__(
            n_frames=n_frames,
            fps=fps,
            inputs=(),
        )

    def get_fps(self):
        return self.fps

    def get_key_indices(self):
        return np.arange(self.stream.frames)

    def read(self, index, force_type=np):

        if index < 0 or index >= self.n_frames:
            raise IndexError(index)

        # Seek one frame earlier for decoder stability
        seek_index = max(0, index - 2)

        timestamp = round(
            (seek_index / self.fps)
            / self.stream.time_base
        )

        self.container.seek(
            timestamp,
            stream=self.stream,
            any_frame=True,
            backward=True
        )

        # Important after random seeks
        self.stream.codec_context.flush_buffers()
        for frame in self.container.decode(self.stream):
            # Estimate frame index from timestamp
            if frame.time is not None:
                current_index = round(
                    frame.time * self.fps
                )

            elif frame.pts is not None:
                current_index = round(
                    frame.pts
                    * float(frame.time_base)
                    * self.fps
                )

            else:
                continue

            if current_index >= index:
                fmt = frame.format.name
                # Preserve native grayscale/RGB + bit depth
                if fmt.startswith("gray") or "rgb" in fmt:
                    arr = frame.to_ndarray()
                else:
                    bits_per_channel = max(c.bits for c in frame.format.components)
                    arr = frame.to_ndarray(format = "rgb48le" if bits_per_channel > 8 else "rgb24")

                # Ensure channel dimension exists
                if arr.ndim == 2:
                    arr = arr[..., None]

                return VideoSupplier.convert(
                    arr,
                    force_type
                )
        raise IndexError(f"Frame {index} not found")

    def get_meta_data(self):
        return {
            "fps": self.get_fps()
        }

    def close(self, recursive=False):
        self.container.close()