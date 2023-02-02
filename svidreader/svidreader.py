import hashlib
import imageio
from ccvtools import rawio


class SVidReader:
    def __init__(self, video, hash_iterator=iter):
        self.video = video
        self.reader = imageio.get_reader(video)
        self.hashes = []
        self.has_issues = False

        # Don't consume self.reader
        hash_reader = FixedLengthReader(imageio.get_reader(video), self.reader.count_frames())
        for img in hash_iterator(hash_reader):
            self.hashes.append(hashlib.md5(img).hexdigest())

    def get_data(self, fr_idx):
        fr_hash = self.hashes[fr_idx]

        requ_idx = fr_idx
        # If we know this file is problematic, start with one frame earlier, as going backwards is expensive
        if self.has_issues and requ_idx >= 1:
            requ_idx = requ_idx-1

        img = self.reader.get_data(requ_idx)
        imghash = hashlib.md5(img).hexdigest()
        if imghash == fr_hash:
            return img
        elif requ_idx >= 1:
            self.has_issues = True
            cur_fr_idx = self.hashes.index(imghash)
            requ_idx = fr_idx + requ_idx - cur_fr_idx
            print(f"SVidReader: Wanted {fr_idx}, tryed {requ_idx}, got {cur_fr_idx}, try {requ_idx}")
            img = self.reader.get_data(requ_idx)
            imghash = hashlib.md5(img).hexdigest()
            if imghash == fr_hash:
                return img
            else:
                raise FrameNotFoundError()
        else:
            raise FrameNotFoundError()

    def get_meta_data(self, fr_idx):
        return self.reader.get_meta_data(fr_idx)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.reader.get_next_data()
        except IndexError:
            raise StopIteration

    def __len__(self):
        return len(self.hashes)


class FixedLengthReader:
    def __init__(self, reader, length):
        self.reader = reader
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.reader.get_next_data()
        except IndexError:
            raise StopIteration

    def __len__(self):
        return self.length


class FrameNotFoundError(Exception):
    pass
