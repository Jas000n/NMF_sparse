import time

import torch
import torchnmf
import numpy as np
from scipy.sparse import csr_matrix

import mv100
def rmse(test_rm, estimate_rm):
    n = test_rm.shape[0]
    m = test_rm.shape[1]
    sum = 0
    total = 0
    for i in range(0, n):
        for j in range(0, m):
            if (test_rm[i][j] != 0):
                sum += (test_rm[i][j] - estimate_rm[i][j]) ** 2
                total += 1
    return (sum / total) ** 0.5
train_list = mv100.mv1002list("./ml-100k/u5_fix.base")
test_list = mv100.mv1002list("./ml-100k/u5.test")
test_rm = mv100.creat_matrix(test_list)
train_rm = mv100.creat_matrix(train_list)

# V = torch.rand(33, 50).unsqueeze(0).unsqueeze(0)
matrix = torch.tensor(train_rm,dtype=float)
model = torchnmf.nmf.NMF(matrix.shape, rank=50)
time1 = time.time()
model.fit(matrix)
w = model.W
h = model.H
print(h.size())
print(w.T.size())
erm = torch.mm(h,w.T).detach().numpy()
print(erm)
time2 = time.time()
print("RMSE loss on test set is {}\n"
      "using {} seconds in total!".format(rmse(test_rm,erm),time2-time1))