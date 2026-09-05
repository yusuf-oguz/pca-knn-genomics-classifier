from pathlib import Path
import pandas as pd
import numpy as np


def load_dataset(data_dir: Path, geno_file: str, lab_file: str):
    # Function to load the dataset from the data directory
    # Input: data_dir - path to the data directory
    #        geno_file - name of the genotype file
    #        lab_file - name of the label file
    # Output: X_df - pandas dataframe of shape (N, M)
    #         y - numpy array of shape (N,)
    #         iids - numpy array of shape (N,)
    #         snp_names - numpy array of shape (M,)

    geno_df = pd.read_csv(data_dir / geno_file, index_col=0)
    lab_df = pd.read_csv(data_dir / lab_file, index_col=0)

    common = geno_df.index.intersection(lab_df.index)
    geno_df = geno_df.loc[common]
    lab_df = lab_df.loc[common]

    X_df = geno_df.replace(-1, np.nan)
    y = lab_df.iloc[:, 0].to_numpy()
    iids = geno_df.index.to_numpy()
    snp_names = geno_df.columns.to_numpy()

    return X_df, y, iids, snp_names
