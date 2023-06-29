import math
import numpy as np

def comp_resolution(y):
    y = y.copy()
    y = np.squeeze(y)
    y.sort()
    y_rolled = np.roll(y, 1)
    res = np.min(np.abs(y - y_rolled))

    if not res:
        res = 10.01

    return res

def score_residuals_sse(sse, n, resolution):
    sigma_sq = (sse / n) ** 2
    score = (n / 2) * (1/np.log(2) + np.log2(2*math.pi*sigma_sq)) - n * logg(resolution)

    return max(0, score)

def score_residuals(residuals):
    residuals = np.squeeze(residuals)

    sse = np.sum(np.power(residuals - np.mean(residuals), 2))
    n = len(residuals)
    resolution = comp_resolution(residuals)

    return score_residuals_sse(sse, n, resolution)

def score_integer(int_num):
    int_num = abs(int_num)

    score = 0
    
    if int_num >= 1:
        log_star = logg(int_num)
        score = log_star

        while log_star > 0:
            log_star = logg(log_star)
            score += log_star

        score += logg(2.865064)
        
    return score


def score_float(float_num):
    int_num = np.ceil(float_num)

    return score_integer(int_num)

def logg(x):
        return 0 if x == 0 else np.log2(x)


def score_weights(weights):
    weights_score = 0
    
    for float_w in weights:
        try:
            float_w = float_w[0]
        except:
            pass

        float_w = abs(float_w)
        if float_w > 1e-12:
            precision = 1
            while float_w < 1000:
                precision += 1
                float_w *= 10
            
            weights_score += score_integer(precision) + score_float(float_w) + 1
        
    
    return weights_score