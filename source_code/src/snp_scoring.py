import numpy as np
import pandas as pd


def pca_correlated_snp_scores(V: np.ndarray, k: int) -> np.ndarray:
    # score each SNP by summing squared loadings over the top k PCs
    scores = np.sum(V[:, :k] ** 2, axis=1)
    return scores


def select_top_snps(snp_names: np.ndarray, snp_scores: np.ndarray, top_k: int = 200):
    # sort SNPs by score and return the top_k as a DataFrame
    idx = np.argsort(-snp_scores)[:top_k]
    top_snps_df = pd.DataFrame({
        "SNP": snp_names[idx],
        "score": snp_scores[idx],
        "rank": np.arange(1, top_k + 1)
    })
    return top_snps_df
