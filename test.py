import time

import numpy as np
from sklearn.cluster import KMeans

import mv100
from scipy.sparse import csr_matrix

train_list = mv100.mv1002list("./ml-100k/u5_fix.base")
test_list = mv100.mv1002list("./ml-100k/u5.test")
test_rm = mv100.creat_matrix(test_list)
train_rm = mv100.creat_matrix(train_list)
print("in the movie-100k datasets, there are {} users and {} items !".format(train_rm.shape[0], train_rm.shape[1]))

for i in range(1, 1683):
    found = False
    for j in test_list:
        if (i == j[1]):
            found=True
            break
    if(found==False):
        print(i)