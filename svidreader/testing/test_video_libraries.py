import unittest
from contextlib import ExitStack
from svidreader import filtergraph
import numpy as np

class TestVideoLibraries(unittest.TestCase):
    def test_write_read_nokeyframe(self):
        filename = "./test/cubes.mp4"
        frames = (0, 200, 30)
        reader_libs = ["iio", "pyav", "decord"]
        with ExitStack() as stack:

            readers = [stack.enter_context(filtergraph.get_reader(filename, backend=lib)) for lib in reader_libs]
            for fr in frames:
                images = [reader.read(fr) for reader in readers]
                for i in range(1, len(images)):
                    np.testing.assert_equal(images[0], images[i])

