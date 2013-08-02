from nose.tools import *

import numpy as np

from wavelets import Wavelets
from wavelets import WaveletAnalysis


N = 1000
x = np.random.random(N)

wa = WaveletAnalysis(x)

def test_N():
    assert_equal(N, wa.N)
