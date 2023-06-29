import numpy as np
from mdl import (
    comp_resolution,
    score_float,
    score_integer,
    score_residuals,
    score_residuals_sse,
    score_weights,
    logg
)

class TestMDLBasics:
    def test_comp_resolution(self):
        y = np.asarray([1, 4, 3])
        resolution = comp_resolution(y)

        assert resolution == 1

        y = np.asarray([0, 0, 0])
        resolution = comp_resolution(y)

        assert resolution == 10.01
    
    def test_score_float(self):
        assert score_float(1.0) == score_integer(1)
        assert score_float(1.1) == score_integer(2)
        assert score_float(1.1) == score_float(1.9)
    
    def test_score_integer(self):
        assert score_integer(0) == 0
        assert score_integer(1) - 1.518 < 0.001
        assert score_integer(2) - 3.518 < 0.001
    
    def test_score_weights(self):
        assert score_weights(np.linspace(1, 10, 1))
    
    def test_logg(self):
        assert logg(0) == 0
        assert logg(1) == 0
        assert logg(2) == 1
    
    def test_score_residuals_sse(self):
        assert score_residuals_sse(1000, 1000, 0.0001) > 0
    
    def test_score_residuals(self):
        assert score_residuals(np.linspace(10, 100, 10)) > 0

