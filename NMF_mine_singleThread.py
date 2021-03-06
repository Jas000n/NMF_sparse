import time
import numpy as np
from scipy.sparse import csr_matrix
import mv100
from scipy.sparse import rand
import plt


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


def updateP(rows, cols, p, q_const, error_matrix):
    i = 0
    for row, col in zip(rows, cols):
        rui = error_matrix.__getitem__((row, col))
        p[row, :] += rui * q_const.getcol(col).transpose() * lr
        i += 1
        if (i % 10000 == 0):
            print("P updating finished {}%!".format(i / 800))


def updateQ(rows, cols, p_const, q, error_matrix):
    i = 0
    for row, col in zip(rows, cols):
        rui = error_matrix.__getitem__((row, col))
        q[:, col] += rui * p_const.getrow(row).transpose() * lr
        i += 1
        if (i % 10000 == 0):
            print("Q updating finished {}%!".format(i / 800))


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
epochs = 300
lr = 0.0001
erm = p.dot(q)  # estimated rating matrix

rows, cols = train_rm.nonzero()
# for epoch in range(0,epochs):

train_loss = []
test_loss = []
x = []

for epoch in range(0, epochs):
    time1 = time.time()
    erm = p.dot(q)  # estimated rating matrix
    error_matrix = train_rm - erm
    p_const = p
    q_const = q
    updateP(rows, cols, p, q_const, error_matrix)
    #time.sleep(1)
    updateQ(rows, cols, p_const, q, error_matrix)
    time2 = time.time()
    print(p * q)
    test_loss_ = rmse(test_rm, p.dot(q))
    train_loss_ = rmse(train_rm, p.dot(q))
    print("this is the {} epoch\n"
          "rmse loss on training set is {}\n"
          "rmse loss on test set is {}\n"
          "for this epoch using {} seconds".format(epoch + 1, train_loss_, test_loss_, time2 - time1))
    x.append(epoch + 1)
    test_loss.append(test_loss_)
    train_loss.append(train_loss_)
    #plt.plot(x, train_loss, test_loss)
    np.save("save_matrix"+str(epoch), np.array(p.dot(q).todense()))