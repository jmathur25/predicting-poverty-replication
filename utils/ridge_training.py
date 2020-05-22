# This is based on code from the Jean et al Github that is modified to work with Python3 and our metrics

import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sklearn.linear_model as linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.cluster import KMeans

def run_randomized_cv(X, y, k=5, k_inner=5, random_seed=7, points=10,
        alpha_low=1, alpha_high=5, to_print=False):
    """
    Run randomized CV on given X and y
    Returns r2, yhat
    """
    np.random.seed(random_seed)
    alphas = np.logspace(alpha_low, alpha_high, points)
    r2s = []
    y_hat = np.zeros_like(y)
    kf = KFold(n_splits=k, shuffle=True)
    fold = 0
    for train_idx, test_idx in kf.split(X):
        if to_print:
            print(f"fold: {fold}", end='\r')
        r2, y_p = evaluate_fold(X, y, train_idx, test_idx, k_inner, alphas, to_print)
        r2s.append(r2)
        y_hat[test_idx] = y_p
        fold += 1
    return np.mean(r2s), y_hat


def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler.
    """
    X_scaler = StandardScaler(with_mean=True, with_std=False)
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    return X_train, X_test


def train_and_predict_ridge(alpha, X_train, y_train, X_test):
    """
    Trains ridge model and predicts test set.
    """
    ridge = linear_model.Ridge(alpha)
    ridge.fit(X_train, y_train)
    y_hat = ridge.predict(X_test)
    return y_hat

def find_best_alpha(X, y, k_inner, alphas, to_print=False):
    """
    Finds the best alpha in an inner fully randomized CV loop.
    """
    kf = KFold(n_splits=k_inner, shuffle=True)
    best_alpha = 0
    best_r2 = 0
    for idx, alpha in enumerate(alphas):
        y_hat = np.zeros_like(y)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, X_test = scale_features(X_train, X_test)
            y_hat[test_idx] = train_and_predict_ridge(alpha, X_train, y_train, X_test)
        r2 = metrics.r2_score(y, y_hat)
        if r2 > best_r2:
            best_alpha = alpha
            best_r2 = r2
    if to_print:
        print(best_alpha)
    return best_alpha


def evaluate_fold(X, y, train_idx, test_idx, k_inner, alphas, to_print=False):
    """
    Evaluates one fold of outer CV.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    best_alpha = find_best_alpha(X_train, y_train, k_inner, alphas, to_print)
    X_train, X_test = scale_features(X_train, X_test)
    y_test_hat = train_and_predict_ridge(best_alpha, X_train, y_train, X_test)
    r2 = metrics.r2_score(y_test, y_test_hat)
    return r2, y_test_hat


def run_spatial_cv(X, y, groups, k_inner=5, random_seed=7, points=10,
        alpha_low=1, alpha_high=5, to_print=False):
    """
    Run randomized CV on given X and y
    Returns r2, yhat
    """
    np.random.seed(random_seed)
    alphas = np.logspace(alpha_low, alpha_high, points)
    k = int(groups.max() + 1)
    r2s = []
    y_hat = np.zeros_like(y)
    fold = 0
    for i in range(k):
        train_idx = groups != i
        test_idx = groups == i
        if to_print:
            print(f"fold: {fold}", end='\r')
        r2, y_p = evaluate_fold(X, y, train_idx, test_idx, k_inner, alphas)
        # could use this function to do inner-fold spatial validation
        # r2, y_p = evaluate_spatial_fold(X, y, groups, train_idx, test_idx, alphas)
        r2s.append(r2)
        y_hat[test_idx] = y_p
        fold += 1
    return np.mean(r2s), y_hat

def evaluate_spatial_fold(X, y, groups, train_idx, test_idx, alphas):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    best_alpha = find_best_alpha_spatial(X_train, y_train, groups_train, alphas)
    X_train, X_test = scale_features(X_train, X_test)
    y_test_hat = train_and_predict_ridge(best_alpha, X_train, y_train, X_test)
    r2 = metrics.r2_score(y_test, y_test_hat)
    return r2, y_test_hat

def find_best_alpha_spatial(X, y, groups, alphas):
    """
    Finds the best alpha in an inner spatial CV loop.
    """
    gs = np.unique(groups)
    best_alpha = 0
    best_r2 = 0
    for alpha in alphas:
        y_hat = np.zeros_like(y)
        for g in gs:
            # hold out each g in the inner spatial loop while choosing the best alpha
            train_idx = groups != g
            test_idx = groups == g
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, X_test = scale_features(X_train, X_test)
            y_hat[test_idx] = train_and_predict_ridge(alpha, X_train, y_train, X_test)
        r2 = metrics.r2_score(y, y_hat)
        if r2 > best_r2:
            best_alpha = alpha
            best_r2 = r2
    return best_alpha

def assign_groups(df, k, random_seed=7):
    ''' Assign clusters in df (columns cluster_lat, cluster_lon) into k groups, also returns cluster centers'''
    np.random.seed(random_seed)
    km = KMeans(k)
    return km.fit_predict(df[['cluster_lat', 'cluster_lon']]), km.cluster_centers_
