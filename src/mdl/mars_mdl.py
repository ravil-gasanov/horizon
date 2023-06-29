import numpy as np
import pandas as pd
from scipy.special import comb
from sklearn.metrics import r2_score

from models import Mars
from mdl import score_residuals_sse, score_integer, score_weights, comp_resolution

class MarsMDL:
    def combinator(self, n_of_parents, n_of_terms):
        sum = comb(n_of_parents + n_of_terms - 1, n_of_parents)
        
        return 0 if sum == 0.0 else np.log2(sum)

    def score_hinges(self, n_of_parents, coeffs, hinges, interactions):
        F = 9
        score = score_integer(hinges)

        for n_of_terms in interactions:
            score += score_integer(n_of_terms) + self.combinator(n_of_parents, n_of_terms) \
                + n_of_terms * np.log2(F)
        
        score += score_weights(coeffs)

        return score

    def mars_score(self, n_of_parents, n_of_datapoints, n_of_vars,\
         sse, coeffs, hinges, interactions, resolution):

            score = score_integer(n_of_parents) + n_of_parents * np.log2(n_of_vars) \
                + self.score_hinges(n_of_parents, coeffs, hinges, interactions)
            
            resid_score = score_residuals_sse(sse, n_of_datapoints, resolution)
            
            # print(f"MARS Model score: {score}")
            # print(f"MARS Residual score: {resid_score}")
            # print(f"MARS mse: {sse/n_of_datapoints}")
            # print(f"MARS comp resol: {resolution}")

            score += resid_score
            
            return score
    
    def score(self, X, y, n_of_vars):
        mars = Mars()
        sse, coeffs, hinges, interactions = mars.fit_predict(X, y)
        n_of_datapoints, n_of_parents = X.shape
        resolution = comp_resolution(y)

        score = self.mars_score(n_of_parents, n_of_datapoints, n_of_vars,\
         sse, coeffs, hinges, interactions, resolution)

        return score