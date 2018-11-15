'''
Tested on: Python 3.5.2, numpy 1.14.1, pandas 0.20.3

Please read README for info about the dataset.

**Please note that all code here is optional -- feel free to
use a completely different implementation.**
'''
import numpy as np
import pandas as pd

np.random.seed(42)

def get_user_data():
    '''Returns user info as a Pandas DataFrame,
    where rows are users and columns are the features
    UserID, Gender, Age, Occupation, Zip Code'''
    users_filename = "users.dat"
    return pd.read_csv(users_filename, header=None, sep='::', engine='python')


def get_movie_data():
    '''Returns movie info as a Pandas DataFrame,
    where rows are movies and columns are the features
    MovieID, Title, Genres'''
    movies_filename = "movies.dat"
    return pd.read_csv(movies_filename, header=None, sep='::', engine='python')


def get_rating_data():
    '''Returns rating info as two 2D numpy arrays:
    one for training and one for testing.
    Rows are ratings and columns are
    UserID, MovieID, Rating, Timestamp'''
    ratings_filename = "ratings.dat"
    df = pd.read_csv(ratings_filename, header=None, sep='::', engine='python')
    data = df.ix[:,:2].values
    np.random.shuffle(data)

    train_length = int(df.shape[0] * .8)

    train = data[:train_length]
    test = data[train_length:]
    return train, test


def get_rating_matrix(df, num_users, num_movies):
    '''Given a Pandas Dataframe containing UserID, MovieID, and Rating,
    returns a 2D numpy array Y where Y[UserID][MovieID] = Rating for all entries
    in df, with all other elements equal to None.'''
    # Y = np.
    pass


def SV_thresholding_hard(data_train, data_test, tau_h=2):
    Y_prime = None # TODO
    U, sigma, V = SVD(Y_prime)

    # TODO: truncate SVs using hard thresholding
    p_hat = None
    A_hat = None 

    # TODO: Evaluate using some error metric measured on test set
    error = None # TODO
    return error


def SV_thresholding_soft(data_train, data_test, tau_s=2):
    Y_prime = None # TODO
    U, sigma, V = SVD(Y_prime)

    # TODO: truncate SVs using soft thresholding
    p_hat = None
    A_hat = None 

    # TODO: Evaluate using some error metric measured on test set
    error = None # TODO
    return error


def SVD(Y):
    pass # TODO: compute SV decomposition of a matrix Y


def ALS(data_train, data_test, k=2, lam=0.02, max_iter=100):
    # size of the problem
    n = max(d[0] for d in data_train)+1 # users
    m = max(d[1] for d in data_train)+1 # items
    # which entries are set in each row and column and the rating
    us_from_v = [[] for i in range(m)]  # II (i-index-set)
    vs_from_u = [[] for a in range(n)]  # AI (a-index set)
    for (a, i, r) in data_train:
        us_from_v[i].append((a, r))
        vs_from_u[a].append((i, r))
    # Initial guesses for u, b_u, v, b_v
    # Note that u and v are lists of column vectors (rows of U, V).
    x = ([np.random.normal(1/k, size=(k,1)) for a in range(n)],
          np.zeros(n),
          [np.random.normal(1/k, size=(k,1)) for i in range(m)],
          np.zeros(m))
    # Alternation, modifies the contents of x
    for i in range(max_iter):
        pass # TODO
    # TODO: Evaluate using some error metric measured on test set
    error = None
    return error


def update_U(data, vs_from_u, x, k, lam):
    pass # TODO


def update_V(data, us_from_v, x, k, lam):
    pass # TODO


def CF_user_user(data_train, data_test, k=1):
    Y_hat = None # TODO

    sim = None # TODO

    # TODO: choose an aggregation method to predict unknown ratings

    # TODO: Evaluate using some error metric measured on test set
    error = None # TODO
    return error


def neural_network(data_train, data_test):
    pass # TODO: use a neural network to predict ratings. Open-ended!


if __name__ == '__main__':
    data_train, data_test = get_rating_data()

    # Singular Value Thresholding
    # SV_thresholding(data_train, data_test)

    # Alternating Least Squares
    # ALS(data_train, data_test)

    # Collaborative Filtering
    # CF_user_user(data_train, data_test)

    # Neural Network
    # neural_network(data_train, data_test)

    # TODO: compare performance of different models
    # TODO determine effect of hyperparameters in each model