import numpy as np
from sklearn.cluster import KMeans

import mv100
from scipy.sparse import csr_matrix

train_list = mv100.mv1002list("./ml-100k/u5_fix.base")
test_list = mv100.mv1002list("./ml-100k/u5.test")
test_rm = mv100.creat_matrix(test_list)
train_rm = mv100.creat_matrix(train_list)
print("in the movie-100k datasets, there are {} users and {} items !".format(train_rm.shape[0], train_rm.shape[1]))

print(train_rm)
kmeans = KMeans(n_clusters=2, random_state=0, max_iter=100000, tol=1e-10).fit(train_rm)
labels = kmeans.labels_
type1 = []
type2 = []
for i in range(labels.shape[0]):
    if (labels[i] == 0):
        type1.append(i)
    else:
        type2.append(i)

type1_rates = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
type2_rates = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
print('type1=', type1)
print('type2=', type2)
sum1 = 0
sum2 = 0
for i in train_list:
    if (i[0] in type1):
        j = int(i[2]) - 1
        type1_rates[j][1] += 1
        sum1 += 1
    else:
        j = int(i[2]) - 1
        type2_rates[j][1] += 1
        sum2 += 1
for i in type1_rates:
    i[1] = i[1] / sum1 * 100
for i in type2_rates:
    i[1] = i[1] / sum2 * 100
type1_rates_ndarray = np.array(type1_rates)
type2_rates_ndarray = np.array(type2_rates)

np.save("./clusters/type1", type1_rates_ndarray)
np.save("./clusters/type2", type2_rates_ndarray)
print(type1_rates_ndarray)
print(type2_rates_ndarray)
print("done!")
