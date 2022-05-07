import get_top_k
import numpy as np

import mv100


matrix = np.load("save_matrix1111.npy",allow_pickle=True)

rated_matrix = mv100.creat_matrix(mv100.mv1002list("/Users/jas0n/PycharmProjects/NMF_sparse/ml-100k/u5_fix.base"))
get_top_k.get_k_ratings(matrix,2)
get_top_k.get_rated_index(rated_matrix,1)