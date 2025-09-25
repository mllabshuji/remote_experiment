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


def main(initial_input, initial_rigidities):

    initial_output = initial_rigidities # こっちも変更予定

    bounds = np.array([[0, 5], [0, 5]])

    # データの取得
    last_index, last_values = _modules.read_csv("analysis_datas.csv")

    if last_index is None:
        raise FileNotFoundError("[MyError]: analysis_datas.csv is not found")
    else:
        # mass = last_values["mass"]
        rigidity = last_values["rigidity"]
        # stiffness = last_values["stiffness"]
    
    training_input = initial_input
    training_output = np.vstack([initial_output, np.array(rigidity)])

    model = _modules.GPmodels(training_input, training_output, kernel_name="RBF", hyparas={"variance":1., "lengthscale":1.}, normalizer=True)
    model.my_optimize()
    current_best = np.min(model.Y)

    next_input, _ = _modules.acq_minimize(_modules.EI, bounds, args=(model, current_best))
    print(f"次の入力値: {next_input}")
    add_data = {"h1": next_input[0], "h2": next_input[1]}
    _modules.add_csv("parameters.csv", add_data)


if __name__ == "__main__":
    # initial_masses = np.array([[30], [15], [50]])
    initial_rigidities = np.array([[100], [105], [120]])
    # initial_stiffnesses = np.array([[80], [75], [60]])

    initial_input = np.array([[5, 3], [2, 5], [4,1], [1,4]])
    main(initial_input, initial_rigidities)

    
    



