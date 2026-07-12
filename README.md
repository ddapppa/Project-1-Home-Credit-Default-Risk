# Credit Risk Scoring — Home Credit Default Risk

## 1. Business Problem

Banyak orang, terutama populasi unbanked atau tanpa riwayat kredit yang cukup, kesulitan mendapatkan akses pinjaman dari lembaga keuangan formal. Populasi ini sering menjadi target pemberi pinjaman yang tidak dapat dipercaya karena minimnya data untuk menilai kelayakan kredit mereka.

Home Credit berupaya memperluas inklusi keuangan bagi populasi ini dengan memberikan pengalaman pinjaman yang aman dan positif. Namun, tantangan utamanya adalah membangun sistem yang dapat:

- Memastikan klien yang mampu membayar tidak ditolak (menghindari kehilangan bisnis dan memperluas akses keuangan).
- Memastikan pinjaman diberikan dengan kalender pokok, jatuh tempo, dan pembayaran yang realistis, sehingga memberdayakan klien untuk sukses secara finansial.
- Meminimalkan risiko gagal bayar (default) yang dapat merugikan perusahaan.

## 2. Data Science Problem

Masalah bisnis ini diterjemahkan menjadi masalah klasifikasi biner: memprediksi probabilitas seorang pemohon akan mengalami kesulitan pembayaran (TARGET = 1) atau melunasi pinjaman dengan baik (TARGET = 0), berdasarkan data aplikasi, riwayat kredit eksternal, dan riwayat transaksi sebelumnya.

Project ini tidak hanya berfokus pada akurasi prediksi, tetapi juga pada interpretability — yaitu kemampuan menjelaskan mengapa seorang pemohon diprediksi berisiko tinggi atau rendah, sehingga hasilnya dapat digunakan sebagai business recommendation yang actionable bagi tim credit risk/underwriting.

## 3. Target Audiens

- Tim Credit Risk / Underwriting — menggunakan model untuk mendukung keputusan approval/rejection pinjaman.
- Manajemen/Stakeholder non-teknis — membutuhkan penjelasan sederhana mengapa suatu keputusan diambil (melalui feature importance dan business recommendation).
- Recruiter/Hiring Manager — sebagai portfolio yang menunjukkan kemampuan end-to-end data science: dari business understanding, EDA, feature engineering, modeling, hingga komunikasi hasil.

## 4. Tujuan Project

1. Membangun model klasifikasi untuk memprediksi probabilitas default pemohon pinjaman.
2. Mengidentifikasi fitur-fitur paling berpengaruh terhadap risiko default (feature importance).
3. Menyusun business recommendation yang dapat dijelaskan kepada stakeholder non-teknis mengenai faktor-faktor yang menyebabkan penolakan pinjaman.

## 5. Success Metrics

| Metrik | Alasan Pemilihan |
|---|---|
| AUC-ROC | Metrik utama, sesuai kriteria evaluasi resmi kompetisi Kaggle; cocok untuk data yang sangat imbalanced antara kelas default dan tidak default. |
| Precision & Recall (kelas minoritas) | Untuk memahami trade-off antara salah menolak pemohon layak vs salah menyetujui pemohon berisiko. |
| Feature Importance (SHAP / permutation importance) | Untuk mendukung sisi interpretability dan business recommendation. |

## 6. Dataset

Dataset yang digunakan adalah Home Credit Default Risk dari kompetisi resmi Kaggle:
https://www.kaggle.com/competitions/home-credit-default-risk/data

| File | Deskripsi |
|---|---|
| application_train.csv / application_test.csv | Data utama aplikasi pinjaman, termasuk kolom target TARGET. |
| bureau.csv | Riwayat kredit klien dari institusi finansial lain. |
| bureau_balance.csv | Data bulanan riwayat kredit dari bureau.csv. |
| previous_application.csv | Riwayat aplikasi pinjaman sebelumnya di Home Credit. |
| POS_CASH_balance.csv | Riwayat bulanan pinjaman POS/cash sebelumnya. |
| credit_card_balance.csv | Riwayat saldo bulanan kartu kredit sebelumnya. |
| installments_payments.csv | Riwayat pembayaran cicilan. |
| HomeCredit_columns_description.csv | Deskripsi lengkap seluruh kolom di semua file. |

## 7. Scope & Constraints

- Tahap awal project akan berfokus pada application_train.csv dan application_test.csv, kemudian dilanjutkan dengan penggabungan tabel lain (bureau, previous_application, dll) pada tahap Feature Engineering.
- Model yang dibangun bersifat dummy project untuk keperluan portfolio, bukan untuk digunakan dalam keputusan kredit nyata.

## 8. Struktur Project

credit-risk-scoring/
README.md
- data/
- raw/
- processed/
- notebooks/
00_problem_definition.md
01_eda.ipynb
02_feature_engineering.ipynb
03_modeling.ipynb
04_evaluation.ipynb
- src/
__init__.py
config.py
dataset.py
cleaning.py
features.py
-- modeling/
__init__.py
train.py
predict.py
plots.py
- models/
- reports/
- figures/
requirements.txt

## 9. Cara Menjalankan Project

git clone <repo-url>
cd credit-risk-scoring
pip install -r requirements.txt

Download dataset dari link Kaggle di atas, letakkan di data/raw/, lalu jalankan notebook secara berurutan mulai dari 01_eda.ipynb.

## 10. Status Project

Sedang dikembangkan — mengikuti Data Life Cycle: Problem Definition → Data Collection → Cleaning → EDA → Feature Engineering → Modeling → Evaluation → Deployment → Documentation.
