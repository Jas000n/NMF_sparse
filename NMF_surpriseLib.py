import time

from surprise import SVD
from surprise import Dataset
from surprise import *
from surprise.model_selection import cross_validate, train_test_split
from surprise import NMF

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.25)
# use NMF algorithm with 300 iterations and k = 50
algo = NMF(n_epochs=300, n_factors=50)
algo.fit(trainset)


# # Run 5-fold cross-validation and print results
# time1 = time.time()
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# time2 = time.time()
# print("using {} seconds!".format(time2-time1))

