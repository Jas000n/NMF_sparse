import numpy as np
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
train_matrix = mv100.creat_matrix(train_list)
X = np.array(train_matrix)
test_list = mv100.mv1002list("./ml-100k/u5.base")
test_matrix = mv100.creat_matrix(test_list)
from sklearn.decomposition import NMF

model = NMF(n_components=50, init='random', random_state=0, max_iter=1000)
W = model.fit_transform(X)
H = model.components_
erm = W.dot(H)
print(type(erm))
print(rmse(test_matrix, erm))
