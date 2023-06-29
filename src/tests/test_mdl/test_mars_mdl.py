import numpy as np
from mdl import MarsMDL, score_residuals
from sklearn.preprocessing import StandardScaler

class TestMarsMDL:
    def test_score(self):
        n = 1000
        X = np.random.normal(0, 1, size=(n, 2))
        y = 0.5 * X.T[0] + 2.5 * X.T[1] + np.random.normal(0, 0.1, n)

        mars_mdl = MarsMDL()

        assert mars_mdl.score(X, y, 3) > 0
    
    def test_reject_null(self):
        n = 1000
        x = np.random.normal(10, 1, size=n)
        y = 0.5 * x**2 + np.random.normal(10, 2, n)

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        mars_mdl = MarsMDL()

        assert mars_mdl.score(x, y, 2) < score_residuals(y)
    
    def test_retain_null(self):
        n = 1000
        x = np.random.normal(0, 1, n)
        y = np.random.normal(0, 2, n)

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        x = StandardScaler().fit_transform(x)
        y = StandardScaler().fit_transform(y)

        mars_mdl = MarsMDL()

        assert mars_mdl.score(x, y, 2) > score_residuals(y)
    
    def test_causal_vs_anti_causal(self):
        n = 1000
        x = 5 + np.random.normal(0, 1, n)
        y = 0.5*np.power(x, 2) + np.random.normal(0, 1, n)

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        x = StandardScaler().fit_transform(x)
        y = StandardScaler().fit_transform(y)

        mars_mdl = MarsMDL()
        causal_score = score_residuals(x) + mars_mdl.score(x, y, 2)
        anti_causal_score = score_residuals(y) + mars_mdl.score(y, x, 2)

        assert causal_score < anti_causal_score
