import numpy as np

import mv100


def rmse(rm, erm):
    sum = 0
    i = 0
    error_matrix = rm - erm
    rows, cols = rm.nonzero()
    for row, col in zip(rows, cols):
        rui = error_matrix.__getitem__((row, col))
        sum += rui ** 2
        i += 1
    return (sum / i) ** 0.5
erm_sup = np.load("sup_model.npy", allow_pickle=True)
rm = mv100.creat_matrix(mv100.mv1002list("./ml-100k/u5_fix.base"))
print(rmse(rm,erm_sup))