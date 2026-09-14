# Home Credit Default Risk — Dokumentasi Data Science

> **Catatan cakupan**: README ini mendokumentasikan seluruh proses Data Science
> project ini, dari perumusan masalah sampai evaluasi model final di holdout
> set. Dokumentasi arsitektur dan cara menjalankan aplikasi Deployment
> (FastAPI) sengaja dipisah ke file terpisah agar dua audiens (evaluator
> proses data science vs pengguna aplikasi) tidak tercampur.

## Daftar Isi

1. [Problem Definition & Business Understanding](#1-problem-definition--business-understanding)
2. [Data Collection](#2-data-collection)
3. [Data Cleaning & Preprocessing](#3-data-cleaning--preprocessing)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Feature Engineering](#5-feature-engineering)
6. [Modeling](#6-modeling)
7. [Evaluation](#7-evaluation)
8. [Ringkasan Keputusan Teknis Kunci](#8-ringkasan-keputusan-teknis-kunci)

---

## 1. Problem Definition & Business Understanding

**Masalah bisnis**: perusahaan pembiayaan (Home Credit) ingin memperluas akses
kredit ke nasabah yang minim atau tanpa riwayat kredit formal, tanpa
menaikkan risiko gagal bayar secara tidak terkendali. Dibutuhkan model yang
bisa memprediksi **probabilitas seorang pemohon akan gagal membayar
pinjaman** (`TARGET = 1`) berdasarkan data aplikasi dan riwayat kredit
eksternalnya.

**Target audiens project**: portofolio untuk melamar posisi Data
Scientist/Analyst, menunjukkan kemampuan menjalankan siklus Data Science
lengkap — bukan cuma melatih model, tapi juga menangani data imbalance,
disiplin anti-kebocoran data (CV, holdout, threshold tuning), dan
interpretabilitas model.

**Definisi sukses**:
- Model punya kemampuan diskriminasi yang jelas lebih baik dari tebakan acak
  (ROC-AUC signifikan di atas 0,5) dan dari baseline sederhana.
- Threshold keputusan didasarkan pada **biaya bisnis asimetris** — di kasus
  credit risk, gagal mendeteksi nasabah yang akan default (*false negative*)
  jauh lebih mahal daripada menolak nasabah baik (*false positive*).
- Model bisa dijelaskan (*explainable*) sampai level fitur dan level individu
  nasabah, bukan cuma "kotak hitam".

**Karakteristik data**: sangat imbalanced — hanya **8,07%** dari seluruh
pemohon yang benar-benar gagal bayar (`TARGET=1`). Ini jadi pertimbangan
utama di hampir setiap tahap: pemilihan metrik (ROC-AUC & PR-AUC, bukan
accuracy), teknik penyeimbangan (`scale_pos_weight`/`class_weight`), dan
pemilihan threshold (bukan default 0,5).

## 2. Data Collection

Dataset publik **Home Credit Default Risk** (kompetisi Kaggle), terdiri dari
tabel utama dan lima tabel riwayat kredit pendukung:

| Tabel | Isi |
|---|---|
| `application_train.csv` | 307.511 baris x 122 kolom — data pengajuan pinjaman + `TARGET` |
| `bureau.csv` | Riwayat kredit nasabah di biro kredit eksternal |
| `bureau_balance.csv` | Riwayat bulanan status kredit bureau |
| `previous_application.csv` | Riwayat pengajuan pinjaman sebelumnya ke Home Credit |
| `POS_CASH_balance.csv` | Riwayat bulanan pinjaman POS/cash sebelumnya |
| `credit_card_balance.csv` | Riwayat bulanan kartu kredit sebelumnya |
| `installments_payments.csv` | Riwayat pembayaran cicilan sebelumnya |

Keenam tabel ini digabung berjenjang (`SK_ID_CURR` → `SK_ID_BUREAU`/`SK_ID_PREV`)
di tahap Feature Engineering untuk menghasilkan fitur agregat per nasabah.

## 3. Data Cleaning & Preprocessing

### 3.1 Tabel Utama (`application_train`)

Proses: `notebooks/02_Cleaning.ipynb`, output: `data/processed/application_train_clean.csv`
(307.511 baris x 101 kolom, dari 91 fitur terpilih di EDA + `SK_ID_CURR` + `TARGET`).

Keputusan cleaning utama, berurutan sesuai notebook:

1. **Sentinel `DAYS_EMPLOYED = 365243`** (55.374 baris, ~18% data) — nilai ini
   bukan masa kerja asli, melainkan penanda "tidak bekerja". Dibuat flag
   `DAYS_EMPLOYED_SENTINEL` untuk menjaga informasinya, lalu nilai asli
   diganti `NaN` dan diimputasi median masa kerja valid (supaya kompatibel
   dengan model linear/MLP yang sensitif terhadap outlier ekstrem).
2. **Hidden missing `XNA`** di `CODE_GENDER`, `NAME_FAMILY_STATUS` — jumlahnya
   sangat kecil (4 dan 2 baris), diimputasi modus.
3. **Grup fitur housing** (area, floor, dst.) — dibuat flag
   `HAS_HOUSING_INFO`, lalu numerik diimputasi median dan kategorikal
   diimputasi kategori `"Missing"` eksplisit (bukan dihapus), karena analisis
   *Information Value* menunjukkan status missing-nya sendiri informatif.
4. **`EXT_SOURCE_1`/`EXT_SOURCE_3`** — diimputasi median per flag missing
   (`EXT_SOURCE_1_MISSING`, `EXT_SOURCE_3_MISSING`), bukan dihapus, karena dua
   fitur ini termasuk prediktor terkuat di seluruh dataset.
5. **`AMT_REQ_CREDIT_BUREAU_*`** — diimputasi 0, dengan asumsi tidak ada
   catatan berarti tidak ada inquiry ke biro kredit.
6. **`OCCUPATION_TYPE`/`ORGANIZATION_TYPE`** — sebelum diimputasi, dilakukan
   verifikasi silang dengan `NAME_INCOME_TYPE`: terbukti 57,46% missing di
   `OCCUPATION_TYPE` dan 100% missing di `ORGANIZATION_TYPE` berasal dari
   nasabah kategori tidak bekerja (Pensioner/Unemployed/Student) — bukan data
   hilang acak. Diimputasi dengan kategori eksplisit `UnknownOccupation` dan
   `NotWorking`.
7. **Log-transform** 4 fitur finansial (`AMT_INCOME_TOTAL`, `AMT_CREDIT`,
   `AMT_ANNUITY`, `AMT_GOODS_PRICE`) — kolom asli tetap disimpan, kolom
   `LOG_*` ditambahkan sebagai alternatif untuk model yang sensitif skala
   (Logistic Regression, MLP).
8. **Konsolidasi kategori langka** (frekuensi < 50) di `NAME_INCOME_TYPE`,
   `ORGANIZATION_TYPE`, `OCCUPATION_TYPE` jadi kategori `"Rare"` — mencegah
   ledakan dimensi saat one-hot encoding.
9. **Validasi akhir**: dipastikan nol missing value tersisa dan konsistensi
   tipe data/kategori (`FLAG_*` cuma 0/1, `CODE_GENDER` cuma M/F).

### 3.2 Lima Tabel Pendukung (Auxiliary Tables)

Diaudit terpisah, keputusan didokumentasikan di
`auxiliary_cleaning_decision_log.json`:

- **Sentinel `365243`** juga muncul di 5 kolom `DAYS_*` pada
  `previous_application` (`DAYS_FIRST_DRAWING` sampai 56,14% terisi nilai
  boneka ini) — diganti `NaN` agar tidak merusak agregasi mean/max nanti.
- **`RATE_INTEREST_PRIMARY`** di-*drop* sepenuhnya (99,65% missing, sinyal
  nyaris tidak ada) — tidak dibawa ke Feature Engineering.
- **Nilai negatif** pada beberapa kolom `AMT_*` (mis. `AMT_CREDIT_SUM_DEBT`,
  `AMT_BALANCE`) — **tidak diapa-apakan**, karena kejadiannya di bawah 0,5%
  dan kemungkinan besar representasi valid dari kelebihan bayar, bukan error.
- **`SK_ID_CURR` yatim** (14-16 baris tidak ditemukan di `application_train`)
  — tidak difilter manual, karena proporsinya konsisten dengan
  `application_test.csv` dan otomatis tersaring saat proses merge berjenjang.
- **`SK_ID_BUREAU`/`SK_ID_PREV` yatim** (38-44 baris) — dikonfirmasi bukan bug
  tipe data lewat verifikasi manual, dibiarkan tersaring otomatis saat merge.
- Tidak ditemukan duplikat baris maupun masalah tipe data di keenam tabel.

## 4. Exploratory Data Analysis

Proses: `notebooks/01_Eda.ipynb`. Dari 122 kolom mentah di `application_train`,
dilakukan seleksi fitur awal berdasarkan analisis missing value, kardinalitas,
dan *Information Value* terhadap `TARGET`, menghasilkan **91 fitur terpilih**
yang menjadi basis input ke tahap Cleaning. Temuan kunci yang mengarahkan
keputusan cleaning: distribusi `TARGET` sangat imbalance (8,07% default),
missing value pada grup fitur housing dan `EXT_SOURCE_1/3` terbukti
informatif (bukan acak), dan missing di `OCCUPATION_TYPE`/`ORGANIZATION_TYPE`
berkorelasi kuat dengan status pekerjaan.

## 5. Feature Engineering

Dilakukan bertahap dalam 3 iterasi, tiap iterasi diuji langsung dampaknya ke
CV ROC-AUC sebelum lanjut ke iterasi berikutnya:

| Versi | Deskripsi | Jumlah fitur (tree) | Dataset |
|---|---|---|---|
| v1 | Fitur dasar dari `application_train_clean` saja | 66 | 307.511 x 73 |
| v2 | + agregasi (`mean/max/min/sum/count`) dari `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `credit_card_balance`, `installments_payments` | ~570-an | — |
| v3 | + 165 fitur rasio custom **recent-window 6 bulan** (`*_RECENT6M_*`) dari tabel bulanan (POS, credit card) | **739** | 307.511 x 746 |

Fitur v3 recent-window dirancang untuk menangkap **tren terbaru** perilaku
nasabah (6 bulan terakhir), bukan cuma agregat sepanjang riwayat — hipotesis
awalnya: perilaku terkini lebih prediktif daripada riwayat lama, dan
terbukti benar (lihat hasil Modeling di bawah).

## 6. Modeling

### 6.1 Baseline (fitur v1)

7 algoritma dibandingkan dengan Stratified 5-Fold CV yang identik:

| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
| Logistic Regression | 0,7475 | 0,2248 |
| Decision Tree | 0,5388 | 0,0912 |
| Random Forest | 0,7441 | 0,2187 |
| LightGBM | 0,7571 | 0,2411 |
| XGBoost | 0,7545 | 0,2388 |
| **CatBoost** | **0,7594** | **0,2436** |
| MLP | 0,7469 | 0,2231 |

CatBoost terpilih sebagai kandidat utama untuk iterasi lanjutan.

### 6.2 Iterasi Fitur (v2 → v3)

| Model | v2 (default) | v2 (tuned) | v3 |
|---|---|---|---|
| CatBoost | 0,7795 | 0,7795 | **0,7829** |
| LightGBM | 0,7775 | — | 0,7810 |
| XGBoost | 0,7733 | — | 0,7777 |

Insight penting: **hyperparameter tuning CatBoost v2 tidak memberi kenaikan
sama sekali** (0,7795 → 0,7795), sementara **fitur baru v3 memberi kenaikan
lebih besar** (+0,0034) — bukti bahwa di titik ini, kualitas dan kuantitas
sinyal (feature engineering) lebih berharga daripada tuning model.

### 6.3 Ensemble dan Stacking

| Pendekatan | ROC-AUC |
|---|---|
| CatBoost v3 sendiri | 0,7829 |
| Ensemble rata-rata sederhana | 0,7834 |
| Ensemble berbobot manual (CatBoost 50% / LightGBM 30% / XGBoost 20%) | 0,7839 |
| Stacking (Logistic Regression di atas OOF, `class_weight="balanced"`) | 0,7841 |

**Keputusan final: CatBoost v3 dipakai sendiri**, bukan ensemble/stacking.
Kenaikan dari ensembling cuma +0,0012 (dalam radius noise CV ±0,002),
sementara kompleksitas deployment naik signifikan (perlu menjalankan 3 model
+ meta-model per prediksi). Trade-off ini tidak sepadan untuk skala project
ini — pertimbangan yang sengaja dipilih untuk menunjukkan *engineering
judgment*, bukan cuma mengejar angka tertinggi.

## 7. Evaluation

### 7.1 Metodologi Anti-Kebocoran Data

- **Holdout split** 80% development / 20% test, stratified berdasarkan
  `TARGET`, dibuat sekali dan di-*freeze* ke `holdout_split.json` — semua
  model memakai sampel identik.
- **Stratified 5-Fold CV** di dalam development set, di-*freeze* ke
  `cv_folds.json`.
- **Threshold keputusan** ditentukan murni dari prediksi *out-of-fold* (OOF)
  hasil CV — **bukan** dari holdout — supaya holdout benar-benar cuma
  "diintip" sekali di akhir untuk laporan final, bukan dipakai berulang untuk
  mencari-cari threshold terbaik.

### 7.2 Pemilihan Threshold (dari OOF)

Tiga metode dibandingkan:

| Metode | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| F1-optimal | 0,651 | 0,264 | 0,449 | 0,333 |
| Recall-optimal (precision floor 30%) | 0,706 | 0,300 | 0,356 | 0,326 |
| **Cost-based (asumsi FN:FP = 5:1)** | **0,6669** | **0,274** | **0,423** | **0,333** |

Rasio biaya **5:1** dipakai sebagai asumsi eksplisit (bukan angka nyata dari
bisnis, karena ini data dummy/publik) — mengikuti rasio yang sama dipakai di
contoh resmi scikit-learn untuk kasus credit scoring: gagal mendeteksi
nasabah gagal bayar (*false negative*) dianggap 5x lebih mahal daripada
menolak nasabah baik (*false positive*).

### 7.3 Hasil Final di Holdout (Sekali, Threshold Terkunci)

Model final: CatBoost v3, dilatih ulang di 100% data development
(246.008 baris), diuji sekali di holdout (61.503 baris, proporsi
`TARGET=1` 8,07%):

| Metrik | CV (OOF) | Holdout |
|---|---|---|
| ROC-AUC | 0,7829 | **0,7862** |
| PR-AUC | 0,2720 | **0,2838** |
| Precision (threshold 0,6669) | 0,274 | 0,279 |
| Recall (threshold 0,6669) | 0,423 | 0,446 |
| F1 | 0,333 | 0,343 |

Holdout **sedikit lebih baik** dari estimasi CV (wajar — model final dilatih
dari 100% data development, bukan 80% seperti tiap fold CV) — tidak ada
tanda overfitting.

Confusion matrix holdout (threshold 0,6669):

| | Prediksi: Baik | Prediksi: Gagal Bayar |
|---|---|---|
| **Aktual: Baik** | 50.811 | 5.727 |
| **Aktual: Gagal Bayar** | 2.750 | 2.215 |

### 7.4 Interpretasi Model (SHAP)

Memakai `shap.TreeExplainer` pada 2.000 sampel dari holdout. Temuan kunci:

- **`EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`** (skor kredit eksternal)
  adalah tiga prediktor terkuat — arah pengaruhnya konsisten dengan intuisi
  bisnis: skor rendah → risiko naik.
- Fitur hasil rekayasa dari riwayat cicilan (`INSTAL_DAYS_PAST_DUE_MAX`,
  `INSTAL_AMT_PAYMENT_SUM`, `INSTAL_PAYMENT_DIFF_MEAN`) masuk 15 fitur
  teratas — **validasi konkret bahwa Feature Engineering v2/v3 memberi nilai
  nyata**, bukan cuma menambah kompleksitas.
- Kontribusi tersebar sangat luas di 725+ fitur lain (gabungan SHAP value-nya
  lebih besar dari 14 fitur teratas digabung) — model mengandalkan kombinasi
  banyak sinyal lemah dari berbagai sumber data, bukan segelintir fitur
  dominan saja.

## 8. Ringkasan Keputusan Teknis Kunci

Ringkasan ini disiapkan untuk menjawab pertanyaan interview "kenapa begini,
bukan begitu":

| Keputusan | Alasan Singkat |
|---|---|
| CatBoost dipilih sebagai basis model | Terbaik di antara 7 algoritma baseline (ROC-AUC 0,7594 vs 0,7475-0,7571 lainnya) |
| Hyperparameter tuning dihentikan lebih awal | Tidak memberi kenaikan (0,7795 → 0,7795); investasi waktu dialihkan ke feature engineering |
| Ensemble/stacking tidak dipakai di final | Kenaikan (+0,0012) tidak sepadan dengan kompleksitas deployment 3x lipat |
| Threshold bukan 0,5 default | Data imbalance 8,07% + biaya kesalahan asimetris (FN jauh lebih mahal dari FP) |
| Threshold ditentukan dari OOF, bukan holdout | Mencegah kebocoran data — holdout harus jadi ujian jujur di akhir, bukan alat tuning |
| CatBoost v3 dipakai sendiri (bukan ensemble) untuk Deployment | Prioritas kesederhanaan dan kemudahan maintain untuk skala project portofolio |

---

*Dokumentasi ini mencakup fase Data Science (Problem Definition s.d.
Evaluation). Lihat dokumentasi terpisah untuk arsitektur dan panduan
menjalankan aplikasi Deployment (FastAPI).*
