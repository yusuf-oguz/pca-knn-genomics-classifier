import numpy as np
import pandas as pd


def filter_missingness(
    X_df: pd.DataFrame,
    y,
    iids,
    snp_names,
    drop_snp_missing: float,
    drop_ind_missing: float,
):
    # drop SNPs that have too many missing values
    snp_miss_rate = X_df.isna().mean(axis=0)
    X2 = X_df.loc[:, snp_miss_rate <= drop_snp_missing]

    # drop individuals that have too many missing values
    ind_miss_rate = X2.isna().mean(axis=1)
    keep = ind_miss_rate <= drop_ind_missing
    keep_arr = keep.to_numpy()
    X2 = X2.loc[keep]
    y2 = y[keep_arr]
    iids2 = iids[keep_arr]
    snp_names2 = X2.columns.to_numpy()

    return X2, y2, iids2, snp_names2


def mean_impute(X_df: pd.DataFrame) -> pd.DataFrame:
    # fill missing values with the column (SNP) mean
    imputed_X_df = X_df.fillna(X_df.mean())
    return imputed_X_df


def center_matrix(X: np.ndarray):
    # subtract column mean from each column
    mu = X.mean(axis=0)
    X_c = X - mu
    return X_c, mu


def mean_impute_using_train(Xtr_df: pd.DataFrame, Xte_df: pd.DataFrame):
    # compute train means and use them for both train and test imputation
    train_means = Xtr_df.mean()
    Xtr_imp = Xtr_df.fillna(train_means)
    Xte_imp = Xte_df.fillna(train_means).fillna(0)
    return Xtr_imp, Xte_imp


def center_apply(X: np.ndarray, mu: np.ndarray):
    # center X using a pre-computed mean
    X_c = X - mu
    return X_c
