import get_top_k
import numpy as np

import mv100


erm = np.load("save_matrix1111.npy",allow_pickle=True)

rm = mv100.creat_matrix(mv100.mv1002list("/Users/jas0n/PycharmProjects/NMF_sparse/ml-100k/u5_fix.base"))
while(1):
    U = eval(input("please input an user id:"))
    get_top_k.get_k_ratings(erm,rm,U,5)
