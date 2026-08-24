# Problem Definition & Business Understanding
## Home Credit Default Risk — Portfolio Project

## 1. Business Problem

Banyak orang, terutama populasi *unbanked* atau tanpa riwayat kredit yang cukup, kesulitan mendapatkan akses pinjaman dari lembaga keuangan formal. Populasi ini sering menjadi target pemberi pinjaman yang tidak dapat dipercaya karena minimnya data untuk menilai kelayakan kredit mereka.

Home Credit berupaya memperluas inklusi keuangan bagi populasi ini dengan memberikan pengalaman pinjaman yang aman dan positif. Namun, tantangan utamanya adalah membangun sistem yang dapat:

- Memastikan klien yang mampu membayar **tidak ditolak** (menghindari kehilangan bisnis dan memperluas akses keuangan).
- Memastikan pinjaman diberikan dengan kalender pokok, jatuh tempo, dan pembayaran yang realistis, sehingga memberdayakan klien untuk sukses secara finansial.
- Meminimalkan **risiko gagal bayar (default)** yang dapat merugikan perusahaan.

### Business Cost Asymmetry

Dua jenis kesalahan model memiliki dampak bisnis yang tidak setara:

| Jenis Kesalahan | Definisi | Dampak Bisnis |
|---|---|---|
| False Negative (FN) | Model prediksi "aman" padahal klien sebenarnya default | **Lebih mahal** — kerugian langsung dari pinjaman yang tidak terbayar |
| False Positive (FP) | Model prediksi "berisiko" padahal klien sebenarnya mampu bayar | Lebih murah — kehilangan potensi bunga/bisnis, tapi tidak ada kerugian modal langsung |

Asimetri ini penting karena akan memengaruhi pemilihan **threshold keputusan** di tahap evaluasi — bukan sekadar threshold default 0.5.

## 2. Data Science Problem

Masalah bisnis ini diterjemahkan menjadi masalah **klasifikasi biner**: memprediksi probabilitas seorang pemohon akan mengalami kesulitan pembayaran (TARGET = 1) atau melunasi pinjaman dengan baik (TARGET = 0), berdasarkan data aplikasi, riwayat kredit eksternal, dan riwayat transaksi sebelumnya.

Project ini tidak hanya berfokus pada akurasi prediksi, tetapi juga pada **interpretability** — yaitu kemampuan menjelaskan mengapa seorang pemohon diprediksi berisiko tinggi atau rendah, sehingga hasilnya dapat digunakan sebagai *business recommendation* yang actionable bagi tim credit risk/underwriting.

### Success Metric

- **Primary metric: AUC-ROC** — metrik evaluasi resmi kompetisi Kaggle Home Credit Default Risk, dipilih karena robust terhadap kelas yang imbalanced (mayoritas klien tidak default) dan mengevaluasi kemampuan model membedakan kelas di seluruh threshold, bukan hanya satu titik potong.
- **Secondary metric: Precision, Recall, dan F1-score pada threshold terpilih** — dibutuhkan karena AUC tidak cukup menjelaskan trade-off bisnis nyata (berapa banyak klien baik yang salah ditolak vs berapa klien default yang lolos). Threshold optimal akan ditentukan mempertimbangkan cost asymmetry di atas.

## 3. Target Audiens

- **Tim Credit Risk / Underwriting** — menggunakan model untuk mendukung keputusan approval/rejection pinjaman.
- **Manajemen/Stakeholder non-teknis** — membutuhkan penjelasan sederhana mengapa suatu keputusan diambil (melalui feature importance dan business recommendation).
- **Recruiter/Hiring Manager** — sebagai portfolio yang menunjukkan kemampuan end-to-end data science: dari business understanding, EDA, feature engineering, modeling, hingga komunikasi hasil.

## 4. Tujuan Project

1. Membangun model klasifikasi untuk memprediksi probabilitas default pemohon pinjaman.
2. Mengidentifikasi fitur-fitur paling berpengaruh terhadap risiko default (feature importance).
3. Menyusun business recommendation yang dapat dijelaskan kepada stakeholder non-teknis mengenai faktor-faktor yang menyebabkan penolakan pinjaman.

## 5. Scope Project (Tahap Awal)

- Dataset utama: `application_train.csv` (memiliki TARGET) digunakan untuk EDA, feature engineering awal, training, dan validasi internal (via train/validation split).
- `application_test.csv` (tanpa TARGET) disiapkan untuk keperluan prediksi akhir/submission, bukan untuk evaluasi performa internal.
- Tabel pendukung (`bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `installments_payments`, `credit_card_balance`) akan diintegrasikan pada tahap Feature Engineering untuk meningkatkan performa model di atas baseline.
