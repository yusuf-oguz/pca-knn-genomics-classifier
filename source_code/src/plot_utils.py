import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_eigenvalues(eigvals: np.ndarray, out_path):
    fig = plt.figure()
    plt.plot(eigvals[:50], marker="o")
    plt.title("Scree Plot (first 50 eigenvalues)")
    plt.xlabel("PC index")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    # plt.close()


def plot_cumulative_variance(eigvals: np.ndarray, out_path):
    evr = eigvals / eigvals.sum()
    cum = np.cumsum(evr)
    plt.figure()
    plt.plot(cum, marker="o")
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of PCs")
    plt.ylabel("Cumulative variance ratio")
    plt.ylim(0, 1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    # plt.close()


def plot_pc_scatter(Z2: np.ndarray, y: np.ndarray, out_path, title: str):
    pops = sorted(set(y.tolist()))
    plt.figure()
    for pop in pops:
        m = y == pop
        plt.scatter(Z2[m, 0], Z2[m, 1], s=18, alpha=0.8, label=pop)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    # plt.close()


def plot_pc_scatter_train_test(
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    Z_te: np.ndarray,
    y_te: np.ndarray,
    out_path,
    title: str = "PC1 vs PC2: Train + Test",
    show_test_labels: bool = True,
):
    """Plot training and test PC scores on the same 2D PCA axes (train=circles, test=crosses).
    If show_test_labels is False, all test points use a single 'Test' label (e.g. for blind evaluation).
    """
    pops_tr = sorted(set(y_tr.tolist()))
    fig = plt.figure()
    for pop in pops_tr:
        m_tr = y_tr == pop
        plt.scatter(
            Z_tr[m_tr, 0], Z_tr[m_tr, 1], s=18, alpha=0.8, label=pop, marker="o"
        )
    if show_test_labels:
        pops_te = sorted(set(y_te.tolist()))
        for pop in pops_te:
            m_te = y_te == pop
            plt.scatter(
                Z_te[m_te, 0],
                Z_te[m_te, 1],
                s=24,
                alpha=0.9,
                marker="x",
                label=f"{pop} (test)",
            )
    else:
        plt.scatter(
            Z_te[:, 0],
            Z_te[:, 1],
            s=24,
            alpha=0.9,
            marker="x",
            label="Test",
        )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path):
    """Plot confusion matrix heatmap from true and predicted labels."""

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, values_format="d"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
