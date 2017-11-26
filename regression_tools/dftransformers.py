import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as SS
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Transformer that selects a column in a numpy array or DataFrame
    by index or name.
    """
    def __init__(self, idxs=None, name=None):
        self.idxs = np.asarray(idxs)
        self.idxs = idxs
        self.name = name

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        # Need to teat pandas data frames and numpy arrays slightly differently.
        if isinstance(X, pd.DataFrame) and self.idxs:
            return X.iloc[:, self.idxs]
        if isinstance(X, pd.DataFrame) and self.name:
            return X[self.name]
        return X[:, self.idxs]


class Identity(TransformerMixin):
    """Transformer that does nothing, simply passes data through unchanged."""
    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        return X


class FeatureUnion(TransformerMixin):
    """Just like sklean.FeatureUnion, but also works for pandas.DataFrame
    objects.
    
    Parameters
    ----------
    transformer_list: list of Transformer objects.
    """
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for _, t in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X, *args, **kwargs):
        Xs = [t.transform(X) for _, t in self.transformer_list]
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return pd.concat(Xs, axis=1)
        return np.hstack(Xs)


class MapFeature(TransformerMixin):
    """Map a function across a feature.

    Parameters
    ----------
    f: function
        The function to map across the array or series.

    name: string
        A name to assign to the transformed series.
    """
    def __init__(self, f, name):
        self.f = f
        self.name = name

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            raise ValueError("You must select a single column of a DataFrame"
                             " before using MapFeature")
        Xind = self.f(X).astype(float)
        if isinstance(X, pd.Series):
            return pd.Series(Xind, index=X.index, name=self.name)
        return Xind

class StandardScaler(TransformerMixin):
    """Standardize all the columns in a np.array or a pd.DataFrame.

    Parameters:
        None
    """
    def __init__(self):
        self._scaler = SS()

    def fit(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            self._scaler.fit(X.values)
        else:
            self._scaler.fit(X)
        return self

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return pd.DataFrame(
                self._scaler.transform(X.values),
                columns=X.columns,
                index=X.index)
        else:
            return self._scaler.transform(X)

class Intercept(TransformerMixin):
    """Create an intercept array or series (containing all values 1.0) of the
    appropriate shape given an array or DataFrame.
    """
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return pd.Series(np.ones(X.shape[0]),
                             index=X.index, name="intercept")
        return np.ones(X.shape[0])
