import numpy as np
import pandas as pd
import warnings
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
    """Encodes dummy variables for all categorical columns in a dataframe.

    The input to this transformer should be an array of strings, integers, or
    floats that denote categorical (discrete) features. The output will match
    the input type (i.e. pandas -> pandas, numpy -> numpy) where each column
    corresponds to a single feature.

    This encoding may be needed for many types of modeling.

    Parameters:
    idxs ("all" or array-like): column indicies to generate dummies
        - array-like str column names for pd.DataFrame
        - array-like int column indicies for np.array
        - "all" (default) for all columns regardless of data object
        - use "all" for pd.Series

    levels (mapping): dict of col_index:list of str or int for levels to
        dummify or None to encode all levels for all features
        important: explicitly defining levels overrides num_dummies,
        i.e. if you want explicitly set the levels with n-1 total levels,
        make sure to do that with this option.

    less (int > 0): for given n feature levels, n - less to encode.
        Useful for models that require n-1 or fewer levels to avoid
        linear algebra solver problems. Use None to encode all levels.

    drop (bool): whether to drop columns that will be dummified.

    Example: df = pd.DataFrame({"color":"blue", "orange", "red"},
                                "flavor": "berry", "orange", "cherry"})
             sd = SmartDummies(levels={"color":["red","blue],
                                       "flavor":["cherry", "berry"]})
             sd.fit(df)
             sd.transform(df)
             >>>"color_red"  "color_blue"  "flavor_cherry"  "flavor_berry"
                 0            1             0                1
                 0            0             0                0
                 1            0             1                0
    """
    def __init__(self, idxs="all", levels=None, less=None, drop=True):
        self.idxs = idxs
        self.levels = levels
        if less == None:
            self.less = None
        else:
            self.less = less
        self.drop = drop

    def fit(self, X):
        """Fit DummiesEncoder to X"""
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
        """Transform X using the dummies encoding"""

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
        """Returns the most common unique levels in array-like arr"""
        levels, counts = np.unique(arr, return_counts=True)
        return levels[np.argsort(counts)[self.less:]]

    def _make_colnames(self, prefix, levels):
        """Returns list of str column names
        Parameters:
        prefix (str): prefix to appear in all column names
        levels (list of str): list of suffixes to append to individual columns

        Returns:
        list of str in form "prefix_level"
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
            dum_arr[:, curr_col_idx:curr_col_idx + len(self.levels[key])] \
                += np.array(self.levels[key]) == curr_vals
            curr_col_idx += len(self.levels[key])
        return dum_arr


class MissingIndicator(TransformerMixin):
    """Returns boolean arrays that indicate missing data.

    The input to this transformer should be array-like objects of any data type
    where missing values are indicated with np.nan. The `fit` method checks for
    columns in the array that are missing data. The `transform` method checks
    for missing data and returns boolean arrays indicating the whether data is
    missing at a given index in the columns that were missing data when fit.

    Attributes
    ----------
    self.missing_data_cols : np.ndarray
        Column indices in X that contain missing data when `.fit(X)` is called
    self.missing_data_names : np.ndarray
        Column names in X that contain missing data when `.fit(X)` is called on
        an instance of pd.DataFrame

    Examples
    --------
    >>> df = pd.DataFrame({"a":[0,1,3,np.nan], "b":[9,np.nan, 3,1],
                           "c":[3,5,2,4]})
    >>> df2 = pd.DataFrame({"a":[3,np.nan, 4], "b":[np.nan, 3,1],
                            "c":[np.nan, 1, 2]})
    >>> mi = MissingIndicator()
    >>> mi.fit(df)
    <MissingIndicator at 0x1a0f08e240>
    >>> mi.transform(df)
      "a_is_missing"  "b_is_missing"
    0 0               0
    1 0               1
    2 0               0
    3 1               0

    >>> mi.transform(df2)
      "a_is_missing"  "b_is_missing"
    0 0               1
    1 1               0
    2 0               0
    """
    def __init__(self):
        """Instantiate MissingIndicator object"""
        self.missing_data_cols = None
        self.missing_data_names = None

    def fit(self, X):
        """Fit MissingIndicator to X"""
        arr = X.copy()
        if isinstance(X, pd.DataFrame):
            arr = arr.values
        self.missing_data_cols = np.where(np.isnan(arr).any(axis=0))[0]
        if isinstance(X, pd.DataFrame):
            self.missing_data_names = X.columns[self.missing_data_cols]
        if len(self.missing_data_cols) == 0:
            warnings.warn("Warning: no missing data found. "
                          "No transformations will be made.")
        return self

    def transform(self, X):
        """Transform X with fit MissingIndicator"""
        if len(self.missing_data_cols) == 0:
            warnings.warn("Warning: no missing data was found during fitting. "
                          "No transformations will be made. Returning X.")
            return X
        missing_array = np.zeros(X.shape, dtype=np.int8)
        missing_array += np.isnan(X)
        # subset columns if necessary
        if len(X.shape) == 2:
            missing_array = missing_array[:,self.missing_data_cols]
        # type checking of X and handling type-specific return requirements
        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                missing_array = missing_array.reshape(-1,1)
            return missing_array
        elif isinstance(X, pd.DataFrame):
            colnames = ([f"{col}_is_missing" for col in self.missing_data_names])
            return pd.DataFrame(missing_array, columns=colnames, index=X.index)
        elif isinstance(X, pd.Series):
            if X.name:
                series_name = str(X.name)+"is_missing"
            else:
                series_name = None
            return pd.Series(missing_array, name=series_name, index=X.index)
        else:
            raise TypeError("Expected pd.DataFrame, pd.Series, or np.ndarray")


class NanReplacer(TransformerMixin):
    """Replaces nans with zeros or a statistic derived from the data

    The input to this transformer should be array-like objects of numeric data
    types where missing data is indicated with np.nan. The `fit` method finds
    the chosen statistic for each column and saves it. The `transform` method
    replaces all np.nan in a column with the corresponding statistic found in
    the fit method.

    Parameters
    ----------
    fill : string
        Value to use for filling missing data.
        - 'mean' (default) : replace missing values with column mean
        - 'zero' : replace missing values with 0
        - 'median' : replace missing values with column median
        - 'max' : replace missing values with column max
        - 'min' : replace missing values with column min

    Attributes
    ----------
    values : array of numeric or numeric
        Values found during fitting to be used to replace missing data during
        transform.

    Examples
    --------
    >>> df = pd.DataFrame({"a":[0,1,3,np.nan], "b":[9,np.nan, 3,1],
                           "c":[3,5,2,4]})
    >>> df2 = pd.DataFrame({"a":[3,np.nan, 4], "b":[np.nan, 3,1],
                            "c":[np.nan, 1, 2]})
    >>> nr = NanReplacer()
    >>> nr.fit(df)
    <NanReplacer at 0x1a0f0953c8>
    >>> nr.transform(df)
      "a"         "b"         "c"
    0 0.000000    9.000000    3.0
    1 1.000000    4.333333    5.0
    2 3.000000    3.000000    2.0
    3 1.333333    1.000000    4.0
    >>> nr.transform(df2)
      "a"         "b"         "c"
    0 3.000000    4.333333    3.5
    1 1.333333    3.000000    1.0
    2 4.000000    1.000000    2.0
    """
    def __init__(self, fill="mean"):
        """Instantiate NanReplacer object"""
        self.fill = fill
        self.values = None

    def fit(self, X):
        """Fit NanReplacer to X"""
        if isinstance(X, pd.Series):
            self.values = self._find_fill(X)
        else:
            self.values = np.apply_along_axis(
                lambda x: self._find_fill (x), 0, X)
        return self

    def transform(self, X):
        """Transform X with fit NanReplacer"""
        num_array = self._replace_nans(X)
        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                num_array = num_array.reshape(-1,1)
            return num_array
        elif isinstance(X, pd.DataFrame):
            colnames = list(X.columns)
            return pd.DataFrame(num_array, columns=colnames, index=X.index)
        elif isinstance(X, pd.Series):
            return pd.Series(num_array, name=X.name, index=X.index)

        else:
            raise TypeError("Expected pd.DataFrame, pd.Series, or np.ndarray")

    def _find_fill(self, arr):
        """Returns the fill value for an array

        Parameters:
        arr (array-like): array of mixed numerical data mixed with nans

        Returns:
        int or float corresponding to user-declared fill method
        """
        if self.fill == "zero":
            return 0
        elif self.fill == "mean":
            return np.nanmean(arr)
        elif self.fill == "max":
            return np.nanmax(arr)
        elif self.fill == "min":
            return np.nanmin(arr)
        elif self.fill == 'median':
            return np.nanmedian(arr)
        else:
            raise ValueError("Use one of ['zero', 'mean', 'median', "
                             "'max', 'min']) for fill")

    def _replace_nans(self, arr):
        """Returns arr with nans replaced with corresponding values in
        self.values

        Parameters:
        arr (array-like): array of mixed numerical data mixed with nans

        Returns:
        np.ndarray of numeric dtype
        """
        arr = arr.copy()
        idxs = np.where(np.isnan(arr))
        # note: two-line if condition below. indentation is confusing.
        if (isinstance(arr, pd.Series) or
            (isinstance(arr, np.ndarray) and len(arr.shape) == 1)):
            arr = np.asarray(arr)
            arr[idxs] = self.values
            return arr
        elif isinstance(arr, pd.DataFrame):
            arr = arr.values
        arr[idxs] = np.take(self.values, idxs[1])
        return arr
