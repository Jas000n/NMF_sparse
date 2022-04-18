import numpy as np
import mv100
import time

train_list = mv100.mv1002list("./ml-100k/u5_fix.base")
train_matrix = mv100.creat_matrix(train_list)
X = np.array(train_matrix)
test_list = mv100.mv1002list("./ml-100k/u5.base")
test_matrix = mv100.creat_matrix(test_list)
from sklearn.decomposition import NMF
import lib_plt


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


x = []
test_loss = []
train_loss = []
for i in range(0, 20000,200):
    time1 = time.time()
    model = NMF(n_components=50, init='random', random_state=0, max_iter=i + 1)
    W = model.fit_transform(X)
    H = model.components_
    print("loss=====================================================")
    test_loss_ = rmse(test_matrix, W.dot(H))
    train_loss_ = rmse(train_matrix, W.dot(H))
    time2 = time.time()
    print("this is {} epoch,\n"
          "the train_loss is {},\n"
          "the test_loss is {}\n"
          "using {} seconds".format(i + 1, train_loss_, test_loss_, time2 - time1))
    x.append(i + 1)
    train_loss.append(train_loss_)
    test_loss.append(test_loss_)
    lib_plt.plot(x, train_loss, test_loss)
