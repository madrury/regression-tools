from math import ceil

import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.base import clone
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt

from basis_expansions.basis_expansions import NaturalCubicSpline


def plot_univariate_smooth(ax, x, y,
    x_lim=None, mask=None, smooth=True, n_knots=6, bootstrap=False):
    """Draw a scatter plot of some (x, y) data, and optionally superimpose
    a cubic spline, or many cubic splines fit to bootstrapped versions of the
    data.

    Parameters
    ----------
    ax: matplotlib.Axis 
        An axis object to draw the plot on.

    x: np.array or pd.Series, shape (n_samples,)
        An np.array or pd.Series object containing the x data.

    y: np.array or pd.Series, shape (n_samples,)
        An np.array or pd.Series object containing the y data.

    x_lim: Tuple of two floats. 
        A tuple contining limits for the x-axis of the plot.  If not supplied,
        this is computed as the minimum and maximum of x.

    mask: np.array or pd.Series of Booleans, shape (n_smaples,) 
        A boolean np.array or pd.Series containing a mask for the x and y data,
        if supplied only the unmasked data contributes to the plot.

    smooth: bool 
        Draw a fit cubic spline or not?

    n_knots: int 
        The number of knots to use in the cubic spline.

    bootstrap: bool or integer
        False or an integer.  The number of times to boostrap the data
        when drawing the spline.  If not False, draw one spline per bootstrap
        sample of the data.

    Returns
    --------
    None
    """
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    if mask is not None:
        if isinstance(mask, pd.Series):
            mask = mask.values
        x = x[mask]
        y = y[mask]
    if not x_lim:
        x_lim = (np.min(x), np.max(x))
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)

    if is_binary_array(y):
        regression_obj = LogisticRegression(
            fit_intercept=True, C=1000, intercept_scaling=1000)
    else:
        regression_obj = LinearRegression(fit_intercept=True)
    
    ax.scatter(x, y, color='grey', alpha=0.25, label="Data")
    if smooth and bootstrap:
        for _ in range(bootstrap):
            x_boot, y_boot = resample(x, y)
            plot_smoother(ax, x_boot, y_boot, x_lim, n_knots, 
                          regression_obj=regression_obj,
                          alpha=0.5, color="lightblue", label=None)        
    if smooth:
        plot_smoother(ax, x, y, x_lim, n_knots, 
                      regression_obj=regression_obj,
                      linewidth=3, color="blue", label="Trend")

def is_binary_array(y):
    return set(np.unique(y)) == {0.0, 1.0}

def plot_smoother(ax, x, y, x_lim, n_knots, regression_obj=LinearRegression,
                  **kwargs):
    """Fit an plot a single cubic spline smoother on a scatterplot."""
    ncr = make_natural_cubic_regression(n_knots, 
                                        regression_obj=regression_obj)
    ncr.fit(x, y)
    t = np.linspace(x_lim[0], x_lim[1], num=250)
    if hasattr(regression_obj, "predict_proba"):
        y_smoothed = ncr.predict_proba(t.reshape(-1, 1))[:, 1]
    else:
        y_smoothed = ncr.predict(t.reshape(-1, 1))
    ax.plot(t, y_smoothed, **kwargs)

def make_natural_cubic_regression(n_knots, regression_obj=LinearRegression, 
                                  knot_range=(-2, 2)):
    """A helper function for constructing a pipeline fiting a one dimensional
    regression with a cubic spline feature."""
    regression_obj = clone(regression_obj)
    return Pipeline([
        ('standardizer', StandardScaler()),
        ('nat_cubic', NaturalCubicSpline(knot_range[0], knot_range[1], n_knots=n_knots)),
        ('regression', regression_obj)
    ])


def display_coef(model, coef_names):
    """Pretty print a table of the parameter estimates in a linear model.

    Parameters
    ----------
    model: A fit sklean object with a `coef_` attribute.

    coef_names: A list of names associated with the coefficients.
    """
    print("{:<35}{:<20}".format("Name", "Parameter Estimate"))
    print("-"*(35 + 20))
    for coef, name in zip(model.coef_, coef_names):
        row = "{:<35}{:<20}".format(name, coef)
        print(row)


def bootstrap_train(model, X, y, bootstraps=1000, **kwargs):
    """Train a (linear) model on multiple bootstrap samples of some data and
    return all of the parameter estimates.

    Parameters
    ----------
    model: A sklearn class whose instances have a `fit` method, and a `coef_`
    attribute.

    X: A two dimensional numpy array of shape (n_observations, n_features).
    
    y: A one dimensional numpy array of shape (n_observations).

    bootstraps: An integer, the number of boostrapped models to train.

    Returns
    -------
    bootstrap_coefs: A (bootstraps, n_features) numpy array.  Each row contains
    the parameter estimates for one trained boostrapped model.
    """
    bootstrap_models = []
    for i in range(bootstraps):
        boot_idxs = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_boot = X[boot_idxs, :]
        y_boot = y[boot_idxs]
        M = model(**kwargs)
        M.fit(X_boot, y_boot)
        bootstrap_models.append(M)
    return bootstrap_models

def get_bootstrap_coefs(bootstrap_models):
    n_models, n_coefs = len(bootstrap_models), len(bootstrap_models[0].coef_)
    bootstrap_coefs = np.empty(shape=(n_models, n_coefs))
    for i, model in enumerate(bootstrap_models):
        bootstrap_coefs[i, :] = model.coef_
    return bootstrap_coefs


def plot_bootstrap_coefs(models, coef_names, n_col=4):
    """Plot histograms of the bootstrapped parameter estimates from a model.
    """
    bootstrap_coefs = get_bootstrap_coefs(models)
    n_coeffs = bootstrap_coefs.shape[1]
    n_row = int(ceil(n_coeffs / n_col)) + 1
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col*6, n_row*2))
    for idx, ax in enumerate(axs.flatten()):
        if idx >= bootstrap_coefs.shape[1]:
            break
        ax.hist(bootstrap_coefs[:, idx], bins=25, color="grey", alpha=0.5)
        ax.set_title(coef_names[idx])
    return fig, axs


def plot_partial_depenence(ax, model, X, var_name,
                           y=None, pipeline=None, n_points=250, **kwargs):
    """Create a partial dependence plot of a feature in a model.

    Parameters
    ----------
    ax: A matplotlib axis object to draw the partial dependence plot on.

    model: A trained sklearn model.  Must implement a `predict` method.

    X: The raw data to use in making predictions when drawing the partial
    dependence plot. Must be a pandas DataFrame.

    var_name: A string, the name of the varaible to make the partial dependence
    plot of.

    y: The y values, only needed if a scatter plot of x vs. y is desired.

    pipeline: A sklearn Pipeline object containing the transformations of the
    raw features used in the model.

    n_points: The number of points to use in the grid when drawing the plot.
    """
    Xpd = make_partial_dependence_data(X, var_name, n_points)
    x_plot = Xpd[var_name]
    if pipeline is not None:
        Xpd = pipeline.transform(Xpd)
    if y is not None:
        ax.scatter(X[var_name], y, color="grey", alpha=0.5)
    y_hat = model.predict(Xpd)
    ax.plot(x_plot, y_hat, **kwargs)

def plot_partial_dependences(model, X, var_names, 
                            y=None, bootstrap_models=None, pipeline=None, n_points=250):
    fig, axs = plt.subplots(len(var_names), figsize=(12, 3*len(var_names)))
    for ax, name in zip(axs, var_names):
        if bootstrap_models:
            for M in bootstrap_models[:100]:
                plot_partial_depenence(
                    ax, M, X=X, var_name=name, pipeline=pipeline, alpha=0.8, 
                    linewidth=1, color="lightblue")
        plot_partial_depenence(ax, model, X=X, var_name=name, y=y,
                               pipeline=pipeline, color="blue", linewidth=3)
        ax.set_title("{} Partial Dependence".format(name))
    return fig, axs

def make_partial_dependence_data(X, var_name, n_points=250):
    Xpd = np.empty((n_points, X.shape[1]))
    Xpd = pd.DataFrame(Xpd, columns=X.columns)
    all_other_var_names = set(X.columns) - {var_name}
    for name in all_other_var_names:
        if is_numeric_array(X[name]):
            Xpd[name] = X[name].mean()
        else:
            # Array is of object type, fill in the mode.
            array_mode = mode(X[name])[0][0]
            Xpd[name] = mode
    min, max = np.min(X[var_name]), np.max(X[var_name])
    Xpd[var_name] = np.linspace(min, max, num=n_points)
    return Xpd

def is_numeric_array(arr):
    """Check if a numpy array contains numeric data.

    Source:
        https://codereview.stackexchange.com/questions/128032
    """
    numerical_dtype_kinds = {'b', # boolean
                             'u', # unsigned integer
                             'i', # signed integer
                             'f', # floats
                             'c'} # complex
    return arr.dtype.kind in numerical_dtype_kinds


def predicteds_vs_actuals(ax, x, y, y_hat, n_bins=50):
    bins, endpoints = pd.cut(x, bins=n_bins, retbins=True)
    centers = (endpoints[:-1] + endpoints[1:]) / 2
    y_hat_means = pd.DataFrame({'y_hat': y_hat, 'bins': bins}).groupby("bins").mean()["y_hat"]
    ax.scatter(x, y, color="grey", alpha=0.5, label="Data")
    ax.scatter(centers, y_hat_means, s=50, label=None)
    ax.plot(centers, y_hat_means, label="Mean Predicted")
    ax.legend()
