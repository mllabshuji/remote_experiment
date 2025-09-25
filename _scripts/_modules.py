import pandas as pd
import os
import numpy as np
import GPy
import scipy
import csv
from scipy.stats import norm
from scipy.optimize import minimize

def read_csv(file_path: str):
    if not os.path.exists(file_path):
        print("読み込みファイルが存在しません")
        return None, None
    else:
        df = pd.read_csv(file_path)
        last_row_index = df.index[-1]
        last_row_values = df.iloc[-1]
        return last_row_index, last_row_values


def add_csv(file_path: str, data: dict):
    if not os.path.exists(file_path):
        print("追加先ファイルが存在しません")
        return None
    else:
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writerow(data)
        return True


class GPmodels(GPy.models.GPRegression):
    def __init__(self, X: np.ndarray, Y: np.ndarray, 
                kernel_name = "RBF", noise_var = 1e-5,
                hyparas: dict = {"variance":1., "lengthscale":1.},
                normalizer=None):
        
        kernel = self.kernel_setting(X.shape[1], kernel_name, hyparas)
        super().__init__(X, Y, kernel, normalizer=normalizer)
        self['Gaussian_noise.variance'].constrain_fixed(noise_var)

        
    def kernel_setting(self, input_dim, kernel_name, hyparas): # kwargs includes hyperparameters
        if kernel_name == "RBF":
            kernel = GPy.kern.RBF(input_dim=input_dim, variance = hyparas["variance"], lengthscale= hyparas["lengthscale"])
            kernel.variance.constrain_fixed(1)
            kernel.lengthscale.constrain_bounded(0, 5)
        elif kernel_name == "SW":
            kernel = GPy.kern.SW_Kernel_ard(input_dim=input_dim, variance = hyparas["variance"], lengthscales= hyparas["lengthscale"])
        return kernel
    
    def my_optimize(self, max_iters=1000):
        super().optimize(max_iters=max_iters, messages=False)
        super().optimize_restarts(num_restarts=10, max_iters=max_iters, messages=False)
    


def EI(X_test: np.ndarray, model: GPmodels, y_best: float, xi=0.01):
    X_test = np.atleast_2d(X_test)  # 単一点でも2次元に
    mu, sigma2 = model.predict(X_test)
    sigma = np.sqrt(sigma2)

    with np.errstate(divide='warn'):
        imp = mu - y_best - xi
        Z = np.zeros_like(imp)
        mask = sigma > 0
        Z[mask] = imp[mask] / sigma[mask]
        ei = np.zeros_like(mu)
        ei[mask] = imp[mask] * norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(Z[mask])

    # 単一点入力ならスカラー、複数点入力なら配列を返す
    if ei.shape[0] == 1:
        return float(ei[0])
    else:
        return ei.flatten()



def acq_minimize(func: callable, bounds: np.ndarray, maximize=True, args=(), method='L-BFGS-B', n_start=10):
    dim = bounds.shape[0]
    if maximize:
        def f(x):
            return -func(x, *args)
    else:
        f = func

    best_val = -np.inf if maximize else np.inf
    best_x = None

    # マルチスタート
    x0s = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_start, dim))
    for x0 in x0s:
        res = minimize(f, x0, bounds=bounds, method=method)
        val = -res.fun if maximize else res.fun
        if (maximize and val > best_val) or (not maximize and val < best_val):
            best_val = val
            best_x = res.x

    return best_x, best_val


