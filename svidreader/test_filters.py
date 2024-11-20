import unittest
from svidreader import effects
import numpy as np


@effects.video_functional
def negative(img):
    return -img

class TestFunctional(unittest.TestCase):
    def test_functional(self):
        array = np.arange(10)
        negative_array = effects.to_array(negative(effects.from_array(array)))
        np.testing.assert_equal(-array, negative_array)

    def test_functional_multithreaded(self):
        array = np.arange(10)
        negative_array = effects.to_array(negative(effects.from_array(array)), jobs=2)
        np.testing.assert_equal(-array, negative_array)