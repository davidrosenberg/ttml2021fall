from sklearn import datasets
import numpy as np


class ClassificationDataGenerator:
    """A wrapper around sklearn's dataaset generator

    sklearn doesn't seem to provide a dataset generator -- only a fixed dataset.
    So our strategy is to generate a very large dataset, and then dish out
    pieces of it on request.  Our strategy here is to sample without replacement
    until the dataset is exhausted, then to "refill the bag" and start over.
    This does not give exactly i.i.d. samples, but should be very close in
    practice when full_n >> n.

    """

    def __init__(
        self, n_features, n_informative, n_redundant, random_state=1, full_n=1000000
    ):
        self.X, self.y = datasets.make_classification(
            n_samples=full_n,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            random_state=random_state,
        )
        self.index = 0
        self.full_n = full_n

    def get_sample(self, n):
        if self.index + n > self.full_n:
            start_X = self.X[self.index :, :]
            start_y = self.y[self.index :]
            # reshuffle data
            indices = np.random.permutation(self.full_n)
            self.X = self.X[indices]
            self.y = self.y[indices]
            self.index = 0
            print("shuffling")
            rest_X, rest_y = self.get_sample(n - len(start_y))
            samp_X = np.vstack((start_X, rest_X))
            samp_y = np.concatenate((start_y, rest_y), axis=0)
        else:
            samp_X = self.X[self.index : (self.index + n), :]
            samp_y = self.y[self.index : (self.index + n)]
            self.index += n

        return samp_X, samp_y
