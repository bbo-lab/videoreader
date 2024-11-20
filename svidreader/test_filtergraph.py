import unittest
from svidreader import filtergraph
import numpy as np
from svidreader.video_supplier import VideoSupplier


class DummyIndexVideo(VideoSupplier):
    def __init__(self, num_frames=10):
        super().__init__(num_frames, ())

    def read(self, index, force_type=np):
        return VideoSupplier.convert(np.full(shape=(1, 1, 1), fill_value=index), force_type)


class TestFilterFunctions(unittest.TestCase):
    def test_named_graph(self):
        with filtergraph.get_reader("./test/cubes.mp4", cache=False) as reader:
            fg = filtergraph.create_filtergraph_from_string([reader],
                                                            '[input_0]cache[cached];[cached]tblend[out]')
            fg['out'].get_data(50)
            print("got", 50)
            for i in range(30, 200):
                fg['out'].get_data(i)

    def test_get_reader(self):
        with filtergraph.get_reader("./test/cubes.mp4|tblend",cache=True) as reader:
            reader.get_data(50)
            print("got",50)
            for i in range(30,200):
                reader.get_data(i)

    def test_brackets(self):
        with filtergraph.get_reader("./test/cubes.mp4") as reader:
            np.testing.assert_equal(reader.read(0), reader[0])


    def test_performance(self):
        import time
        with filtergraph.get_reader("./test/cubes.mp4|analyze", cache=True, options={'lib':'np'}) as reader:
            reader.get_data(0)
            starttime = time.time()
            for i in range(1, 301):
                reader.get_data(i)
            print("ran at ", 300 / (time.time() - starttime), "fps")

            starttime = time.time()
            for frame in reader:
                reader.get_data(i)
            print("ran at ", len(reader) / (time.time() - starttime), "fps")

    def test_iterator(self):
        reader = DummyIndexVideo(num_frames=10)
        import itertools
        for (i_frame, frame) in enumerate(itertools.islice(reader, 2, 5)):
            assert np.all(reader.get_data(i_frame + 2) == frame)

    def test_permutation(self):
        reader = filtergraph.create_filtergraph_from_string([DummyIndexVideo(num_frames=10)],
                                                            "[input_0]permutate=map=./test/test_permutation.csv")['out']
        assert reader.n_frames == 8
        assert reader.read(5).flatten()[0] == 4

        reader = filtergraph.create_filtergraph_from_string([DummyIndexVideo(num_frames=10)],
                                                            "[input_0]permutate=map=./test/test_permutation.csv:sourceoffset=2")['out']
        assert reader.n_frames == 10
        assert reader.read(4).flatten()[0] == 5

        reader = filtergraph.create_filtergraph_from_string([DummyIndexVideo(num_frames=10)],
                                                            "[input_0]permutate=map=./test/test_permutation.csv:sourceoffset=2:destinationoffset=1")['out']
        assert reader.n_frames == 10
        assert reader.read(4).flatten()[0] == 6


if __name__ == '__main__':
    unittest.main()
