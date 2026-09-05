# PCA-Based Ancestry Analysis via SNPs

İTÜ BBF202E (Numerical Methods in Computer Engineering) dönem projesi. İnsan genetik popülasyon kökenini, ham SNP (Single Nucleotide Polymorphism) verisinden, **sıfırdan yazılmış PCA** ve k-NN sınıflandırma ile tahmin eden bir pipeline.

## Problem

4 farklı popülasyondan bireylerin genetik verisi (5000 SNP özelliği) kullanılarak, bir bireyin hangi popülasyona ait olduğunu tahmin etmek:

- **CHB** — Han Chinese in Beijing
- **JPT** — Japanese in Tokyo
- **TSI** — Toscans in Italy
- **YRI** — Yoruba in Nigeria

Veri seti: 256 eğitim + 64 test bireyi.

## Yöntem

1. **Ön işleme** — eksik veri oranına göre SNP/birey filtreleme, SNP-bazında ortalama ile imputation, merkezleme (centering).
2. **PCA — sıfırdan implementasyon** — `numpy.linalg.eig`/`svd` gibi hazır fonksiyonlar kullanılmadı. Özdeğer ayrıştırması, **QR algoritması (QR iteration)** elle yazılarak yapıldı: modified Gram-Schmidt ile QR ayrıştırması → tekrarlı `R@Q` güncellemesiyle yakınsama.
3. **SNP skorlama** — her SNP'nin ilk birkaç temel bileşene (PC) katkısı üzerinden en ayırt edici SNP'ler belirlendi; 10/20/50/100/200 SNP'lik panellerle PCA tekrarlanıp popülasyon ayrışması görselleştirildi.
4. **Doğru train/test metodolojisi** — imputation ortalaması ve PCA projeksiyonu **sadece eğitim verisiyle** hesaplandı, test verisi bu sabit parametrelerle dönüştürüldü (veri sızıntısı yok).
5. **Sınıflandırma** — en ayırt edici 20 SNP'nin PCA uzayında, sıfırdan yazılmış Öklid k-NN (k=5, çoğunluk oyu + eşitlik bozma mantığı) ile test bireyleri sınıflandırıldı.

## Sonuç

**Test doğruluğu: %73.44** (k-NN, k=5, en ayırt edici 20 SNP, 4 sınıflı popülasyon tahmini).

5000 ham SNP arasından sadece 20'sinin seçilmesi, PC1-PC2 düzleminde 4 popülasyonun görsel olarak net şekilde ayrıştığını gösterdi (bkz. `out_dir/pc_scatter_top_20.png`).

## Klasör Yapısı

```
numeric_term project/
├── PCA-Based_Ancestry_Analysis_via_SNPs.pdf   # orijinal ödev tanımı (kendi çalışması değil)
└── source_code/2526SpringBBF202EProject_Student/
    ├── main.ipynb          # uçtan uca pipeline (yükleme → PCA → skorlama → sınıflandırma)
    ├── src/                # modüler kaynak kod
    │   ├── io_utils.py         # veri okuma
    │   ├── preprocess.py       # filtreleme, imputation, merkezleme
    │   ├── pca_ref.py           # PCA — QR algoritmasıyla sıfırdan özdeğer ayrıştırması
    │   ├── snp_scoring.py       # SNP önem skorlama
    │   ├── knn.py                # sıfırdan k-NN sınıflandırıcı
    │   └── plot_utils.py         # görselleştirme
    ├── data/                # genotip ve etiket CSV'leri (train + test)
    ├── out_dir/              # üretilen grafikler, log, sonuç CSV'si
    └── report.pdf / report.tex  # yazılı proje raporu
```

## Kullanılan Araçlar

Python — NumPy (sayısal hesaplama), Pandas (veri işleme), Matplotlib (görselleştirme). Hazır PCA/eigendecomposition kütüphane fonksiyonu **kullanılmadı** — amaç, sayısal yöntemin kendisini implemente etmekti.
