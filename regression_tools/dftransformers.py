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
    """Just like sklearn.FeatureUnion, but also works for pandas.DataFrame
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
        if isinstance(X, pd.DataFrame):
            self._scaler.fit(X.values)
        elif isinstance(X, pd.Series):
            self._scaler.fit(X.values.reshape(-1,1))
        else:
            self._scaler.fit(X)
        return self

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                self._scaler.transform(X.values),
                columns=X.columns,
                index=X.index)
        elif isinstance(X, pd.Series):
            return pd.Series(
                #StandardScaler requires 2-d data, pd.Series requires 1-d data
                self._scaler.transform(X.values.reshape(-1,1)).reshape(-1),
                name=X.name,
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

class DummiesEncoder(TransformerMixin):
    """Makes dummy variables for all categorical columns in a dataframe.
    """
    def __init__(self, idxs="all", levels=None, less=None, drop=True):
        """Instatiates transformer
        Parameters:
        idxs ("all" or array-like): column indicies to generate dummies
            array-like str column names for pd.DataFrame
            array-like int column indicies for np.array
            or "all" for all columns regardless of data object
            use "all" for pd.Series

        levels (mapping): dict of col_index:list of str or int for levels to
            dummify or None to encode all levels for all features
            Example: df = pd.DataFrame({"color":"blue", "orange", "red"},
                                        "flavor": "berry", "orange", "cherry"})
                     sd = SmartDummies(levels={"color":["red","blue],
                                               "flavor":["cherry", "berry"]})
                     sd.fit(df)
                     sd.transform(df)
                     >>>"color_red" "color_blue" "flavor_cherry" "flavor_berry"
                         0           1             0              1
                         0           0             0              0
                         1           0             1              0
            important: explicitly defining levels overrides num_dummies below,
            i.e. if you want explicitly set the levels with n-1 total levels,
            make sure to do that with this option.
        less (int > 0): for given n feature levels, n - less to encode.
            Useful for models that require n-1 or fewer levels to avoid
            gradient descent problems. Use None for all levels.

        drop (bool): whether to drop columns that will be dummified
        """
        self.idxs = idxs
        self.levels = levels
        if less == None:
            self.less = None
        else:
            self.less = -(less)
        self.drop = drop
    def fit(self, X):
        """
        """
        # get all indices if not user-defined
        if self.idxs == "all":
            if isinstance(X, pd.DataFrame):
                self.idxs = [col for col in X.columns]
            elif isinstance(X, pd.Series):
                self.idxs = [X.name]
            elif isinstance(X, np.ndarray):
                self.idxs = [i for i in range(X.shape[1])]
            else:
                raise TypeError("Expected pd.DataFrame, pd.Series, or np.ndarray")

        # get levels if not user-defined
        if not self.levels:
            self.levels = {}
            for index in self.idxs:
                if isinstance(X, pd.DataFrame):
                    self.levels[index] = self._get_levels(X[index])
                elif isinstance(X, pd.Series):
                    self.levels[index] = self._get_levels(X)
                else:
                    self.levels[index] = self._get_levels(X[:,index])
        return self
    def transform(self, X):
        """
        """

        # handle pd.DataFrame
        if isinstance(X, pd.DataFrame):
            colnames = []
            dum_arr = self._make_dum_array(X)
            for key in self.levels:
                colnames.extend(self._make_colnames(key, self.levels[key]))
            dum_df = pd.DataFrame(dum_arr, columns=colnames)
            if self.drop:
                temp_df = X[[col for col in X.columns if col not in self.idxs]]
                return pd.concat((temp_df, dum_df), axis=1)
            else:
                return pd.concat((X, dum_df), axis=1)
        # handle pd.Series
        elif isinstance(X, pd.Series):
            colnames = []
            values = np.array(self.levels[self.idxs[0]]) == X.values.reshape(-1,1)
            if not self.drop:
                values = np.concatenate((X.values.reshape(-1,1), values), axis=1)
                colnames.append(X.name)
                colnames.extend(self._make_colnames(X.name,
                                                    self.levels[self.idxs[0]]))
            else:
                colnames = self._make_colnames(X.name, self.levels[self.idxs[0]])
            return pd.DataFrame(values, columns=colnames)

        # handle np.array
        # check for correct dimensions here?
        elif isinstance(X, np.ndarray):
            # instantiate empty array of zeros to update with dummies
            dum_arr = self._make_dum_array(X)

            if self.drop:
                non_dum_arr = X[:,[i for i in range(X.shape[1])
                                   if i not in self.idxs]]
                return np.concatenate((non_dum_arr, dum_arr), axis=1)
            else:
                return np.concatenate((X, dum_arr), axis=1)

        else:
            raise TypeError("Expected pd.DataFrame, pd.Series, or np.ndarray")


    def _get_levels(self, arr):
        """Returns unique levels in array-like arr"""
        return list(set(arr))[:self.less]

    def _make_colnames(self, prefix, levels):
        """Returns list of str column names
        Parameters:
        prefix (str): prefix to appear in all column names
        levels (list of str): list of suffixes to append to individual columns

        Returns:
        list of str in form prefix_level
        """
        return [f"{prefix}_{level}" for level in levels]

    def _make_dum_array(self, X):
        """Returns np.array of dummied variables"""
        dum_col_cnt = sum(len(val_list) for val_list in self.levels.values())
        dum_arr = np.zeros((X.shape[0], dum_col_cnt), dtype=np.int8)
        curr_col_idx = 0
        for key in self.levels:
            if isinstance(X, pd.DataFrame):
                curr_vals = np.array(X[key].values).reshape(-1,1)
            else:
                curr_vals = X[:,key].reshape(-1,1)
            # broadcast boolean check to fill array
            (dum_arr[:,curr_col_idx:curr_col_idx + len(self.levels[key])]) += (np.array(self.levels[key]) == curr_vals)
            curr_col_idx += len(self.levels[key])
        return dum_arr
