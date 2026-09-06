# PCA-Based Ancestry Analysis via SNPs

<details>
<summary>🇹🇷 Türkçe özet için tıklayın</summary>

İnsan genetik popülasyon kökenini, ham SNP (Single Nucleotide Polymorphism) verisinden, **sıfırdan yazılmış PCA** ve k-NN sınıflandırma ile tahmin eden bir pipeline.

**Problem:** 4 popülasyondan (CHB, JPT, TSI, YRI) bireylerin 5000 SNP özellikli genetik verisi kullanılarak, bir bireyin hangi popülasyona ait olduğunu tahmin etmek. 256 eğitim + 64 test bireyi.

**Yöntem:** eksik veriye göre filtreleme ve SNP-bazında ortalama imputation; PCA'nın kendisi `numpy.linalg.eig`/`svd` gibi hazır fonksiyonlar kullanılmadan, **QR algoritmasıyla elle yazılan özdeğer ayrıştırması** ile yapıldı; her SNP'nin ilk birkaç temel bileşene katkısı üzerinden en ayırt edici SNP'ler belirlendi; imputation ortalaması ve PCA projeksiyonu sadece eğitim verisiyle hesaplanıp veri sızıntısı önlendi; en ayırt edici 20 SNP'nin PCA uzayında sıfırdan yazılmış Öklid k-NN (k=5) ile sınıflandırma yapıldı.

**Sonuç:** %73.44 test doğruluğu. 5000 ham SNP arasından sadece 20'sinin seçilmesi, PC1-PC2 düzleminde 4 popülasyonun görsel olarak net şekilde ayrıştığını gösterdi.

</details>

---

A pipeline that predicts human genetic population ancestry from raw SNP (Single Nucleotide Polymorphism) data, using PCA and k-NN classification built entirely from scratch.

## Problem

Given genetic data (5000 SNP features) for individuals from four populations, predict which population an individual belongs to:

- **CHB**: Han Chinese in Beijing
- **JPT**: Japanese in Tokyo
- **TSI**: Toscans in Italy
- **YRI**: Yoruba in Nigeria

The dataset has 256 training individuals and 64 test individuals.

## Method

1. **Preprocessing.** SNPs and individuals are filtered based on missing-data rate, missing values are imputed with the per-SNP mean, and the data is centered.
2. **PCA, implemented from scratch.** No `numpy.linalg.eig` or `svd`. Eigendecomposition is done by hand with the QR algorithm: a QR decomposition via modified Gram-Schmidt, applied repeatedly through an `R @ Q` update until convergence.
3. **SNP scoring.** Each SNP's contribution to the first few principal components determines which SNPs are the most discriminative. PCA is then re-run on panels of the top 10, 20, 50, 100, and 200 SNPs to see how population separation changes as the panel shrinks.
4. **Correct train/test methodology.** The imputation means and the PCA projection are computed on the training data only, and the test data is transformed using those fixed parameters, so there's no data leakage.
5. **Classification.** Test individuals are classified in the PCA space of the top 20 most discriminative SNPs, using a from-scratch Euclidean k-NN (k=5, majority vote with tie-breaking logic).

## Result

**73.44% test accuracy** (k-NN, k=5, top 20 SNPs, 4-way population classification).

Cutting 5000 raw SNPs down to just 20 still left the four populations visually well separated on the PC1-PC2 plane (see `out_dir/pc_scatter_top_20.png`).

## Folder structure

```
pca-knn-genomics-classifier/
├── PCA-Based_Ancestry_Analysis_via_SNPs.pdf   The original problem statement (not my own work)
└── source_code/
    ├── main.ipynb           The end-to-end pipeline: load, PCA, scoring, classification
    ├── src/                  Modular source code
    │   ├── io_utils.py          Data loading
    │   ├── preprocess.py        Filtering, imputation, centering
    │   ├── pca_ref.py            PCA, eigendecomposition from scratch via the QR algorithm
    │   ├── snp_scoring.py        SNP importance scoring
    │   ├── knn.py                 k-NN classifier, from scratch
    │   └── plot_utils.py          Plotting
    ├── data/                 Genotype and label CSVs, train and test
    ├── out_dir/               Generated plots, logs, and result CSV
    └── report.pdf / report.tex   The written report
```

## Tools

Python, NumPy for the numerical work, Pandas for data handling, Matplotlib for plotting. No ready-made PCA or eigendecomposition library function was used; the point was implementing the numerical method itself.
