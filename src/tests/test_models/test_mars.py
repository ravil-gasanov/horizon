import numpy as np
import pandas as pd
from models import Mars

class TestMars:
    def test_fit(self):
        mars = Mars()
        n = 1000
        x = np.random.normal(10, 1, size=n)
        y = 0.5 * x**5 + np.random.normal(0, 0.1, n)
        
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)        

        print(x.shape)
        print(y.shape)

        sse, _, _, _ = mars.fit_predict(x, y)

        null_sse = float(np.sum(np.power(y - np.mean(y), 2)))

        assert sse < null_sse