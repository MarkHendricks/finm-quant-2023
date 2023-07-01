#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------
# FINM 25000 - UTILITY FUNCTIONS
# ------------------------------
# Disclaimer & Usage Notes:
# ------------------------------------------------------------------------------
# While a decent amount of effort has gone into ensuring correctness,
# there is no guarantee that these functions are 100% correct. You are
# free to use these functions, without attribution, in your own work.
#
# The general structure in terms of naming conventions is that any function
# beginning with "calc_", is a function that calculates something, and will
# return a DataFrame or a dict. Any function beginning with "plot_", is a
# function that plots something, and will return a matplotlib axes object.
#
# Any function that begins with "print_", is a function that prints something
# and returns nothing.
# ------------------------------------------------------------------------------


import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# --------------------
# REGRESSION FUNCTIONS
# --------------------


def calc_univariate_regression(y, X, intercept=True, adj=12):
    """
    Calculate a univariate regression of y on X. Note that both X and y
    need to be one-dimensional.

    Args:
        y : target variable
        X : independent variable
        intercept (bool, optional): Fit the regression with an intercept or not. Defaults to True.
        adj (int, optional): What to adjust the returns by. Defaults to 12.

    Returns:
        DataFrame: Summary of regression results
    """
    X_down = X[y < 0]
    y_down = y[y < 0]
    if intercept:
        X = sm.add_constant(X)
        X_down = sm.add_constant(X_down)

    model = sm.OLS(y, X, missing="drop")
    results = model.fit()

    inter = results.params[0] if intercept else 0
    beta = results.params[1] if intercept else results.params[0]

    summary = dict()

    summary["Alpha"] = inter * adj
    summary["Beta"] = beta
    down_mod = sm.OLS(y_down, X_down, missing="drop").fit()
    summary["Downside Beta"] = down_mod.params[1] if intercept else down_mod.params[0]

    summary["R-Squared"] = results.rsquared
    summary["Treynor Ratio"] = (y.mean() / beta) * adj
    summary["Information Ratio"] = (inter / results.resid.std()) * np.sqrt(adj)
    summary["Tracking Error"] = (
        inter / summary["Information Ratio"]
        if intercept
        else results.resid.std() * np.sqrt(adj)
    )

    return pd.DataFrame(summary, index=[y.name])


def calc_multivariate_regression(y, X, intercept=True, adj=12):
    """
    Calculate a multivariate regression of y on X. Adds useful metrics such
    as the Information Ratio and Tracking Error. Note that we can't calculate
    Treynor Ratio or Downside Beta here.

    Args:
        y : target variable
        X : independent variables
        intercept (bool, optional): Defaults to True.
        adj (int, optional): Annualization factor. Defaults to 12.

    Returns:
        DataFrame: Summary of regression results
    """
    if intercept:
        X = sm.add_constant(X)

    model = sm.OLS(y, X, missing="drop")
    results = model.fit()
    summary = dict()

    inter = results.params[0] if intercept else 0
    betas = results.params[1:] if intercept else results.params

    summary["Alpha"] = inter * adj
    summary["R-Squared"] = results.rsquared

    X_cols = X.columns[1:] if intercept else X.columns

    for i, col in enumerate(X_cols):
        summary[f"{col} Beta"] = betas[i]

    summary["Information Ratio"] = (inter / results.resid.std()) * np.sqrt(adj)
    summary["Tracking Error"] = (
        inter / summary["Information Ratio"]
        if intercept
        else results.resid.std() * np.sqrt(adj)
    )
    return pd.DataFrame(summary, index=[y.name])


def calc_iterative_regression(y, X, intercept=True, one_to_many=False, adj=12):
    """
    Iterative regression for checking one X column against many different y columns,
    or vice versa. "one_to_many=True" means that we are checking one X column against many
    y columns, and "one_to_many=False" means that we are checking many X columns against a
    single y column.

    To enforce dynamic behavior in terms of regressors and regressands, we
    check that BOTH X and y are DataFrames

    Args:
        y : Target variable(s)
        X : Independent variable(s)
        intercept (bool, optional): Defaults to True.
        one_to_many (bool, optional): Which way to run the regression. Defaults to False.
        adj (int, optional): Annualization. Defaults to 12.

    Returns:
        DataFrame : Summary of regression results.
    """

    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
        raise TypeError("X and y must both be DataFrames.")

    if one_to_many:
        if isinstance(X, pd.Series) or X.shape[1] > 1:
            summary = pd.concat(
                [
                    calc_multivariate_regression(y[col], X, intercept, adj)
                    for col in y.columns
                ],
                axis=0,
            )
        else:
            summary = pd.concat(
                [
                    calc_univariate_regression(y[col], X, intercept, adj)
                    for col in y.columns
                ],
                axis=0,
            )
        summary.index = y.columns
        return summary
    else:
        summary = pd.concat(
            [
                calc_univariate_regression(y, X[col], intercept, adj)
                for col in X.columns
            ],
            axis=0,
        )
        summary.index = X.columns
        return summary


# -----------------------
# RISK & RETURN FUNCTIONS
# -----------------------


def calc_return_metrics(data, as_df=False, adj=12):
    """
    Calculate return metrics for a DataFrame of assets.

    Args:
        data (pd.DataFrame): DataFrame of asset returns.
        as_df (bool, optional): Return a DF or a dict. Defaults to False (return a dict).
        adj (int, optional): Annualization. Defaults to 12.

    Returns:
        Union[dict, DataFrame]: Dict or DataFrame of return metrics.
    """
    summary = dict()
    summary["Annualized Return"] = data.mean() * adj
    summary["Annualized Volatility"] = data.std() * np.sqrt(adj)
    summary["Annualized Sharpe Ratio"] = (
        summary["Annualized Return"] / summary["Annualized Volatility"]
    )
    summary["Annualized Sortino Ratio"] = summary["Annualized Return"] / (
        data[data < 0].std() * np.sqrt(adj)
    )
    return pd.DataFrame(summary, index=data.columns) if as_df else summary


def calc_risk_metrics(data, as_df=False, adj=12):
    """
    Calculate risk metrics for a DataFrame of assets.

    Args:
        data (pd.DataFrame): DataFrame of asset returns.
        as_df (bool, optional): Return a DF or a dict. Defaults to False.
        adj (int, optional): Annualizatin. Defaults to 12.

    Returns:
        Union[dict, DataFrame]: Dict or DataFrame of risk metrics.
    """
    summary = dict()
    summary["Skewness"] = data.skew()
    summary["Excess Kurtosis"] = data.kurtosis()
    summary["VaR (0.05)"] = data.quantile(0.05, axis=0)
    summary["CVaR (0.05)"] = data[data <= data.quantile(0.05, axis=0)].mean()
    summary["Min"] = data.min()
    summary["Max"] = data.max()

    wealth_index = 1000 * (1 + data).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    summary["Max Drawdown"] = drawdowns.min()

    summary["Bottom"] = drawdowns.idxmin()
    summary["Peak"] = previous_peaks.idxmax()

    recovery_date = []
    for col in wealth_index.columns:
        prev_max = previous_peaks[col][: drawdowns[col].idxmin()].max()
        recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin() :]]).T
        recovery_date.append(
            recovery_wealth[recovery_wealth[col] >= prev_max].index.min()
        )
    summary["Recovery"] = ["-" if pd.isnull(i) else i for i in recovery_date]

    summary["Duration (days)"] = [
        (i - j).days if i != "-" else "-"
        for i, j in zip(summary["Recovery"], summary["Bottom"])
    ]

    return pd.DataFrame(summary, index=data.columns) if as_df else summary


def calc_performance_metrics(data, adj=12):
    """
    Aggregating function for calculating performance metrics. Returns both
    risk and performance metrics.

    Args:
        data (pd.DataFrame): DataFrame of asset returns.
        adj (int, optional): Annualization. Defaults to 12.

    Returns:
        DataFrame: DataFrame of performance metrics.
    """
    summary = {**calc_return_metrics(data, adj), **calc_risk_metrics(data, adj)}
    summary["Calmar Ratio"] = summary["Annualized Return"] / abs(
        summary["Max Drawdown"]
    )
    return pd.DataFrame(summary, index=data.columns)


# -----------------------
# CORRELATION HELPERS
# -----------------------


def plot_correlation_matrix(corrs):
    # Correlation helper function.
    return sns.heatmap(
        corrs,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.7,
        annot_kws={"size": 10},
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.75},
    )


def print_max_min_correlation(corrs):
    # Correlation helper function.
    corr_series = corrs.unstack()
    corr_series = corr_series[corr_series != 1]

    max_corr = corr_series.abs().agg(["idxmax", "max"]).T
    min_corr = corr_series.abs().agg(["idxmin", "min"]).T
    min_corr_raw = corr_series.agg(["idxmin", "min"]).T
    max_corr, max_corr_val = max_corr["idxmax"], max_corr["max"]
    min_corr, min_corr_val = min_corr["idxmin"], min_corr["min"]
    min_corr_raw, min_corr_raw_val = min_corr_raw["idxmin"], min_corr_raw["min"]

    print(
        f"Max Corr (by absolute value): {max_corr[0]} and {max_corr[1]} with a correlation of {max_corr_val:.2f}"
    )
    print(
        f"Min Corr (by absolute value): {min_corr[0]} and {min_corr[1]} with a correlation of {min_corr_val:.2f}"
    )
    print(
        f"Min Corr (raw): {min_corr_raw[0]} and {min_corr_raw[1]} with a correlation of {min_corr_raw_val:.2f}"
    )


# -------------------
# PORTFOLIO FUNCTIONS
# -------------------


def calc_tangency_portfolio(mean_rets, cov_matrix):
    """
    Function to calculate tangency portfolio weights. Comes from the
    formula seen in class (Week 1).

    Args:
        mean_rets: Vector of mean returns.
        cov_matrix: Covariance matrix of returns.

    Returns:
        Vector of tangency portfolio weights.
    """
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(mean_rets.shape)
    return (inv_cov @ mean_rets) / (ones.T @ inv_cov @ mean_rets)


def calc_gmv_portfolio(cov_matrix):
    """
    Function to calculate the weights of the global minimum variance portfolio.

    Args:
        cov_matrix : Covariance matrix of returns.

    Returns:
        Vector of GMV portfolio weights.
    """
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except TypeError:
        cov_inv = np.linalg.inv(np.array(cov_matrix))

    one_vector = np.ones(len(cov_matrix.index))
    return cov_inv @ one_vector / (one_vector @ cov_inv @ (one_vector))


def calc_mv_portfolio(mean_rets, cov_matrix, target=None):
    """
    Function to calculate the weights of the mean-variance portfolio. If
    target is not specified, then the function will return the tangency portfolio.
    If target is specified, then we return the MV-efficient portfolio with the target
    return.

    Args:
        mean_rets : Vector of mean returns.
        cov_matrix : Covariance matrix of returns.
        target (optional):  Target mean return. Defaults to None. Note: must be adjusted for
                            annualization the same time-frequency as the mean returns. If the
                            mean returns are monthly, the target must be monthly as well.

    Returns:
        Vector of MV portfolio weights.
    """
    w_tan = calc_tangency_portfolio(mean_rets, cov_matrix)

    if target is None:
        return w_tan

    w_gmv = calc_gmv_portfolio(cov_matrix)
    delta = (target - mean_rets @ w_gmv) / (mean_rets @ w_tan - mean_rets @ w_gmv)
    return delta * w_tan + (1 - delta) * w_gmv


# -----------------------------------------------
# MISC. FUNCTIONS
#
# These are functions that you, as a student, will
# not use, but are used to illustrate some of the
# concepts in the class, via TA code.
# -----------------------------------------------
# TBD.

if __name__ == "__main__":
    print("Loaded utils.py")
