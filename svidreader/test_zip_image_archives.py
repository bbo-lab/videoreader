import unittest
from svidreader import filtergraph
import tempfile
import numpy as np

class TestZipImageArchives(unittest.TestCase):
    def test_write_read_nokeyframe(self):
        with tempfile.NamedTemporaryFile(suffix=".zip") as file:
            with filtergraph.get_reader(f"./test/cubes.mp4|dump=output={file.name}", cache=True) as reader:
                images = [reader.read(i) for i in range(20)]

            with filtergraph.get_reader(f"{file.name}") as reader:
                for i in range(20):
                    np.testing.assert_equal(images[i], reader.read(i))