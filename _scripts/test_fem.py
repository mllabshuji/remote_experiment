import os
os.environ["PYKEOPS_NO_CUDA"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import GPy 
import pandas as pd
import _scripts._modules as _modules


def fem_simulation(h1, h2): # ダミー
    rigidity = np.sin(h1) + np.cos(h2) + np.random.normal(0, 1)*0.1
    return rigidity


def main():
    # データの取得
    last_index, last_values = _modules.read_csv("_datas/parameters.csv")

    if last_index is None:
        raise FileNotFoundError("[MyError]: parameters.csv is not found")
    else:
        h1 = last_values["h1"]
        h2 = last_values["h2"]

    rigidities = fem_simulation(h1, h2) # 計算したとして,,,

    print(f"解析結果: rigidity = {rigidities}")
    add_data = {"rigidity": rigidities}
    print(os.getcwd())
    _modules.add_csv("_datas/analysis_datas.csv", add_data)


if __name__ == "__main__":
    main()
    