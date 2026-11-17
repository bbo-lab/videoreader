from svidreader.video_supplier import VideoSupplier

class ConvertToType(VideoSupplier):
    def __init__(self, reader, target_type=None, target_module=None):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        #get numpy target type from string
        if isinstance(target_module, str):
            if target_module == "numpy":
                import numpy as np
                target_module = np
            elif target_module == "cupy":
                import cupy as cp
                target_module = cp
            else:
                raise ValueError(f"Unknown target_module {target_module}, expected 'numpy' or 'cupy'")

        self.target_type = target_type
        self.target_module = target_module

    def read(self, index, force_type=None):
        assert force_type is None, f"force_type has to be None, got {force_type}"
        frame = self.inputs[0].read(index=index, force_type=None)
        target_module = self.target_module
        if target_module is None and force_type is not None:
            target_module = force_type

        if target_module is not None:
            frame = VideoSupplier.convert(frame, target_module)
        else:
            target_module = VideoSupplier.get_array_module(frame)

        target_type = None
        if isinstance(self.target_type, str):
            if self.target_type == "float":
                target_type = target_module.float32
            elif self.target_type == "double":
                target_type = target_module.float64
            else:
                target_type = target_module.dtype(self.target_type).type

        if target_type is not None:
            if issubclass(target_type, target_module.integer):
                info = target_module.iinfo(target_type)
                frame = target_module.clip(frame, info.min, info.max)
        if target_type is not None:
            frame = frame.astype(target_type, copy=False)
        return frame