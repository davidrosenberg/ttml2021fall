
import abc
import pandas as pd
import numpy as np
from numpy.random import default_rng

class BaseSampler:

    def __init__(self, rng=default_rng()):
        self.rng = rng

    def sample(self, n):
        sample = Sample()

        cov_names = self.get_x_colnames()
        x = self.get_x(n)
        sample.add_columns(x, cov_names, 'x')

        t_name = self.get_t_colnames()
        t_prob_name = self.get_t_prob_colnames()
        t_prob = self.get_t_prob(x)
        sample.add_columns(t_prob, t_prob_name, 't_prob')

        t = self.get_t(t_prob)
        sample.add_columns(t, t_name, 't')

        y_names = self.get_y_colnames()
        y = self.get_y(x, t)
        sample.add_columns(y, y_names, 'y')

        return sample.df

    @abc.abstractmethod
    def get_x(self, n):
        pass

    @abc.abstractmethod
    def get_t_prob(self, x):
        pass

    def get_t(self, t_prob):
        return self.rng.random(t_prob.shape[0]) <= t_prob

    @abc.abstractmethod
    def get_y(self, x, t):
        pass

    def get_x_colnames(self):
        return ['x']

    def get_y_colnames(self):
        return ['y']

    def get_t_colnames(self):
        return ['t']

    def get_t_prob_colnames(self):
        return ['t_prob']


class Sample:
    def __init__(self):
        self._df = pd.DataFrame()

    def add_columns(self, data, col_names, att_name=None):
        """

        Args:
            data: sample data for columns
            col_names: names for the data columns
            att_name: attribute name to access this data from Sample object.
                data can be accessed through Sample().att_name. col_names can be
                accessed through Sample().{att_name}_colnames

        Returns:
            None (just adds columns to df)

        Example:
            s = Sample() then
            s.add_columns(data, colnames, 'x') will add `data` to a
            pandas df with column names `colnames`. We can then access this info
            through s.x and s.x_colnames
        """

        if att_name:
            setattr(self, att_name, data)
            setattr(self, f'{att_name}_colnames', col_names)

        if data is None:
            return

        if data.ndim == 1:
            data = np.reshape(data, (-1, 1))

        if data.shape[1] != len(col_names):
            raise ValueError(
                f'Length of col names [{col_names}] does not match width of '
                f'data [{data.shape[1]}]'
            )
        for i, name in enumerate(col_names):
            self._df[name] = data[:, i]

    @property
    def df(self):
        return self._df