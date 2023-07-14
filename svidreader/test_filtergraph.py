import unittest
import svidreader.filtergraph as filtergraph
import numpy as np

class TestStringMethods(unittest.TestCase):
    def test_named_graph(self):
        fg = filtergraph.create_filtergraph_from_string([],'reader=input=./test/cubes.mp4[vid];[vid]cache[cached];[cached]tblend[out]')
        fg['out'].get_data(50)
        print("got",50)
        for i in range(30,200):
            fg['out'].get_data(i)


if __name__ == '__main__':
    unittest.main()
