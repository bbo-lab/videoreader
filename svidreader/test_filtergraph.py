import unittest
import svidreader.filtergraph as filtergraph
import numpy as np

class TestStringMethods(unittest.TestCase):
    def test_named_graph(self):
    	filtergraph.create_filtergraph_from_string([],'reader=input=./test/cubes.mp4[vid];[vid]cache[cached];[cached]tblend[out]')


if __name__ == '__main__':
    unittest.main()
