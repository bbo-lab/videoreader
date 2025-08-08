import hashlib
from copy import deepcopy

import imageio.v3 as iio
import imageio.v2 as iio2
import numpy as np
from svidreader.video_supplier import VideoSupplier
from svidreader.imagecache import ImageCache


class SVidReader(VideoSupplier):
    def __init__(self, video, calc_hashes=False, hash_iterator=iter, cache=None):
        video = str(video)

        self.video = video
        self.vprops = []
        self.hashes = []
        self.last_frame = -1
        self.has_issues = False
        self.plugin = "pyav"
        self.hash = None

        if video[-4:] == '.ccv':
            self.plugin = None

        pipe = self.video.find("|")
        if pipe >= 0:
            self.video = self.video[0:pipe]
        self.vprops = iio.improps(self.video, plugin=self.plugin)
        self.mdata = iio.immeta(self.video, plugin=self.plugin, exclude_applied=False)
        self.reader = iio.imopen(self.video, "r", plugin=self.plugin)
        if video[-4:] == '.ccv':
            iio2_reader = iio2.get_reader(self.video)
            self.reader.n_frames = len(iio2_reader)
            self.mdata = iio2_reader.get_meta_data()
            self.mdata["sensor"] = dict(self.mdata["sensor"])
            for key in ["offset", "size"]:
                self.mdata["sensor"][key] = list(self.mdata["sensor"][key])
            del self.mdata["sensor"]["_io"]
        else:
           self.reader.n_frames = self.vprops.shape[0]
        if cache is None:
            self.reader.get_key_indices = lambda : None
            self.reader = ImageCache(self.reader, maxcount=500)
        elif cache != False:
            cache.inputs = (reader,)
            self.reader = cache

        if pipe >= 0:
            import svidreader.filtergraph as filtergraph
            self.reader = filtergraph.create_filtergraph_from_string([self.reader],video[pipe+1:len(video)])['out']

        if calc_hashes:
            for img in hash_iterator(iio.imiter(video,
                                                plugin=self.plugin,
                                                thread_type="FRAME",
                                                thread_count=16
                                                ), total=self.reader.n_frames):
                self.hashes.append(hashlib.md5(img).hexdigest())
        self.n_frames = self.reader.n_frames
        self.mdata['num_frames'] = self.n_frames

        super().__init__(n_frames=self.n_frames, inputs=(self.reader,))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self, recursive=False):
        self.reader.close()

    def get_data(self, fr_idx):
        requ_idx = fr_idx
        # If we know this file is problematic, start with one frame earlier, as going backwards is expensive
        if self.has_issues and requ_idx >= 1 and self.last_frame + 1 != fr_idx:
            requ_idx = requ_idx - 1

        img = self.reader.read(index=requ_idx)

        if len(self.hashes) == 0:
            return img

        fr_hash = self.hashes[fr_idx]
        imghash = hashlib.md5(img).hexdigest()

        tries = 0
        while imghash != fr_hash:
            cur_fr_idx = self.hashes.index(imghash)
            print(f"SVidReader: Wanted {fr_idx}, tryed {requ_idx}, got {cur_fr_idx},", end="")
            self.has_issues = True
            if tries > 5:
                print("quitting")
                raise FrameNotFoundError()
            tries += 1

            requ_idx = requ_idx + fr_idx - cur_fr_idx
            print(f"try {requ_idx}")

            img = self.reader.read(index=requ_idx)
            imghash = hashlib.md5(img).hexdigest()

        return img

    def __hash__(self, fast_unsafe=False):
        # Currently, this function is just used to get hash which is used as src_id of the video.
        block_size = 65536
        hash_fun = hashlib.md5()

        if fast_unsafe:
            read_max = block_size * 100
            read_bytes = 0
            with open(self.video, 'rb') as f:
                buffer = f.read(block_size)
                while len(buffer) > 0 and read_bytes < read_max:  # arbitrary count limit
                    hash_fun.update(buffer[0:min(len(buffer), read_max - read_bytes)])
                    read_bytes += len(buffer)
                    buffer = f.read(block_size)
            return int(hash_fun.hexdigest(), 16)

        with open(self.video, 'rb') as f:
            if self.hash is None:
                buffer = f.read(block_size)
                while len(buffer) > 0 :  # arbitrary count limit
                    hash_fun.update(buffer)
                    buffer = f.read(block_size)
                self.hash = int(hash_fun.hexdigest(), 16)
            return self.hash

    def read(self, index, force_type=np):
        tmp = self.get_data(fr_idx = index)
        if len(tmp.shape) == 2:
            tmp = tmp[:,:,np.newaxis]
        return VideoSupplier.convert(tmp, force_type)

    def improps(self):
        return deepcopy(self.vprops)

    def get_meta_data(self):
        return deepcopy(self.mdata)

    def __len__(self):
        return self.n_frames

    def get_key_indices(self):
        return None

class FrameNotFoundError(Exception):
    pass
