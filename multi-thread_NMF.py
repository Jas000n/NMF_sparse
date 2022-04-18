import _thread
import time
import numpy as np
from scipy.sparse import csr_matrix
import mv100
from scipy.sparse import rand


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


def updateP(rows, cols, p, q_const, error_matrix,flag):
    i = 0
    for row, col in zip(rows, cols):
        rui = error_matrix.__getitem__((row, col))
        p[row, :] += rui * q_const.getcol(col).transpose() * lr
        i += 1
        if (i % 10000 == 0):
            print("P updating finished {}%!".format(i/800))
    flag = True
def updateQ(rows, cols, p_const, q, error_matrix,flag):
    i = 0
    for row, col in zip(rows, cols):
        rui = error_matrix.__getitem__((row, col))
        q[:, col] += rui * p_const.getrow(row).transpose() * lr
        i += 1
        if (i % 10000 == 0):
            print("Q updating finished {}%!".format(i/800))
    flag = True

train_list = mv100.mv1002list("./ml-100k/u5_fix.base")
test_list = mv100.mv1002list("./ml-100k/u5.test")
test_rm = csr_matrix(mv100.creat_matrix(test_list))
train_rm = csr_matrix(mv100.creat_matrix(train_list))
print("in the movie-100k datasets, there are {} users and {} items !".format(train_rm.shape[0], train_rm.shape[1]))

m = train_rm.shape[0]  # the numbers of user
n = train_rm.shape[1]  # the numbers of item
k = 50  # Hyper parameters

p = csr_matrix(np.full((m, k), (3 / k) ** 0.5))  # the first matrix
q = csr_matrix(np.full((k, n), (3 / k) ** 0.5))  # the second matrix
epochs = 30
lr = 0.0001
erm = p.dot(q)  # estimated rating matrix

rows, cols = train_rm.nonzero()
# for epoch in range(0,epochs):



for epoch in range(0, epochs):
    threadP = False
    threadQ = False
    time1 = time.time()
    erm = p.dot(q)  # estimated rating matrix
    error_matrix = train_rm - erm
    p_const = p
    q_const = q
    _thread.start_new_thread(updateP,(rows,cols,p,q_const,error_matrix,threadP))
    _thread.start_new_thread(updateQ,(rows,cols,p_const,q,error_matrix,threadQ))
    time2 = time.time()
    time.sleep(10000)
    # print(p * q)
    # print("rmse loss on training set is {}\n"
    #       "rmse loss on test set is {}\n"
    #       "for this epoch using {} seconds".format(rmse(train_rm, p.dot(q)), rmse(test_rm, p.dot(q)), time2 - time1))
