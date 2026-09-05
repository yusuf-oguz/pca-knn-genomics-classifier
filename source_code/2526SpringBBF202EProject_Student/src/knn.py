import numpy as np
import pandas as pd


def knn_predict(Z_train: np.ndarray, y_train: np.ndarray, Z_test: np.ndarray, k: int):
    """
    Euclidean KNN with majority vote; ties broken by closest neighbor among tied classes.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, Z_train.shape[0])

    y_pred = []
    classes = np.unique(y_train)

    for z in Z_test:
        d2 = np.sum((Z_train - z) ** 2, axis=1)  # squared distances
        nn_idx = np.argpartition(d2, k - 1)[:k]
        nn_labels = y_train[nn_idx]

        # majority vote
        counts = {c: 0 for c in classes}
        for lab in nn_labels:
            counts[lab] += 1
        max_count = max(counts.values())
        tied = [c for c, cnt in counts.items() if cnt == max_count]

        if len(tied) == 1:
            y_pred.append(tied[0])
        else:
            # tie-break: choose class of closest neighbor among tied classes
            nn_sorted = nn_idx[np.argsort(d2[nn_idx])]
            chosen = None
            for idx in nn_sorted:
                if y_train[idx] in tied:
                    chosen = y_train[idx]
                    break
            y_pred.append(chosen if chosen is not None else tied[0])

    return np.array(y_pred)


def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray):
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    mat = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for t, p in zip(y_true, y_pred):
        mat.loc[t, p] += 1
    return mat
