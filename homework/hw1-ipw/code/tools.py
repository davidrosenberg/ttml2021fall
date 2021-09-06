
import numpy as np
import pandas as pd
from scipy.special import expit
from backend import BaseSampler

class KangSchafSampler(BaseSampler):
    """
    Named for authors of "Demystifying Double Robustness" (Kang and Schafer)
    x is derived by transforming z from N(0, I_{4x4}) nonlinearly.

    This sampler produces a sample of size n that satisfies (approximately) the
    following properties:
        * mean of y = 210
        * mean of observed y = 200
        * mean of missing y = 220
        * response rate = 0.5
        * Two sets of 4 covariates:
            * (z1, z2, z3, z4) = samples from N(0, I_{4x4}) which are used to define outcomes pi and y
            * (x1, x2, x3, x4) = nonlinearly transformed z covs s.t. "real" analyst would
                have no way of knowing true covariates z. See transform_z below for transformation.
        * Outcomes
            * y = 210 + 27.4z1 + 13.7z2 + 13.7z3 + 13.7z4 + error where error is N(0,1)
            * pi = expit(-z1 + 0.5z2 - 0.25z3 - 0.1z4)
        * very good models of y and pi using z_i
        * good but not great models of y and pi using x_i

    In paper, Kang & Schafer typically use n = 200 and n = 1000
    """
    N_COV = 4  # num of covariates
    BASE_COV_NAMES = [f'z{i + 1}' for i in range(N_COV)]
    COV_NAMES = [f'x{i + 1}' for i in range(N_COV)]
    PROP_WEIGHTS = np.array([-1, 0.5, -0.25, -0.1])  # for propensity score function
    Y_WEIGHTS = np.array([27.4, 13.7, 13.7, 13.7])   # for response function
    Y_BIAS = 210                                     # for response function

    def get_x(self, n):
        # function (to be applied to rows of nx4 np.arr) that returns KangSchaf's covariates
        transform_z = lambda z: (
            np.exp(z[0] / 2),                # x1 = exp(z1/2)
            z[1] / (1 + np.exp(z[0])) + 10,  # x2 = z2/(1 + exp(z1)) + 10
            (z[0] * z[2] / 25 + 0.6) ** 3,   # x3 = (z1*z3/25 + 0.6)^3
            (z[1] + z[3] + 20) ** 2          # x4 = (z2 + z4 + 20)^2
        )

        # generate samples
        z = self.rng.normal(size=[n, self.N_COV])
        covs = np.apply_along_axis(func1d=transform_z, axis=1, arr=z)
        return np.column_stack([z, covs])

    def get_x_colnames(self):
        return self.BASE_COV_NAMES + self.COV_NAMES

    def get_t_prob(self, x):
        w = np.append(self.PROP_WEIGHTS, np.array([0, 0, 0, 0]))
        return expit(np.dot(x, w))

    def get_t_prob_colnames(self):
        return ['obs_prob']

    def get_t_colnames(self):
        return ['obs']

    def get_y(self, x, t):
        w = np.append(self.Y_WEIGHTS, np.array([0, 0, 0, 0]))
        return self.rng.normal(loc=np.dot(x, w) + self.Y_BIAS)

def get_estimator_stats(estimates, true_mean=None):
    """
    Given list of estimates for various estimators, consolidate into summary performance
    stats of each estimator

    Args:
        estimates (pd.DataFrame): each row corresponds to collection of estimates for a sample and
            each column corresponds to an estimator
        true_mean (float): the true population mean (if there was no missing data) which will be
            applied to all columns. Note: if including an estimator for a different estimand
             that does not have

    Returns:
        pd.Dataframe where each row represents data about a single estimator
    """
    est_stat = []
    for est in estimates.columns:
        pred_means = estimates[est]
        stat = {}
        stat['stat'] = est
        stat['mean'] = np.mean(pred_means)
        stat['SD'] = np.std(pred_means)
        stat['SE'] = np.std(pred_means) / np.sqrt(len(pred_means))
        if true_mean:
            stat['bias'] = stat['mean'] - true_mean
            stat['RMSE'] = np.sqrt(np.mean((pred_means - true_mean) ** 2))
#            stat['MAE'] = np.mean(abs(pred_means - true_mean))
        est_stat.append(stat)

    return pd.DataFrame(est_stat)
