from surprise import SVD
from surprise import Dataset
from surprise import *
from surprise.model_selection import cross_validate
from surprise import SVDpp


# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

# We'll use the famous SVD algorithm.
algo = SVDpp()

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)