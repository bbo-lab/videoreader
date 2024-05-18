import unittest
from svidreader import filtergraph
import tempfile
import numpy as np

class TestZipImageArchives(unittest.TestCase):
    def test_write_read_nokeyframe(self):
        with tempfile.NamedTemporaryFile(suffix=".zip") as file:
            reader = filtergraph.get_reader(f"./test/cubes.mp4|dump=output={file.name}", cache=True)
            images = [reader.read(i) for i in range(20)]
            reader.close()
            reader = filtergraph.get_reader(f"{file.name}")
            for i in range(20):
                np.testing.assert_equal(images[i], reader.read(i))