# Explainability: Bagaimana Model Menjelaskan Keputusannya

## 1. Apa Itu SHAP (Untuk Non-Teknis)

Model machine learning sering disebut "kotak hitam" — dia bisa memberi skor risiko,
tapi tidak menjelaskan *kenapa*. **SHAP (SHapley Additive exPlanations)** adalah teknik
yang membuka kotak hitam itu: untuk setiap prediksi, SHAP menghitung seberapa besar
kontribusi masing-masing fitur (misal penghasilan, skor eksternal, riwayat kredit)
terhadap skor akhir — apakah fitur itu **menaikkan** atau **menurunkan** risiko, dan
seberapa besar pengaruhnya.

Analoginya seperti dokter yang tidak cuma bilang "Anda berisiko sakit jantung 70%",
tapi juga menjelaskan "karena tekanan darah tinggi (+20%), riwayat keluarga (+15%),
tapi pola makan sehat (-10%)". SHAP memberi rincian semacam itu untuk model kredit ini.

## 2. Implementasi di Project Ini

| Aspek | Detail |
|---|---|
| Metode | `shap.TreeExplainer` (dioptimalkan untuk model tree-based seperti CatBoost) |
| Sampel referensi | 2.000 sampel dari holdout set |
| Top predictor global | `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1` |
| Jumlah faktor ditampilkan per prediksi | Top 5 (`TOP_N_REASONS = 5`) |
| Filter minimum | Kontribusi ≥ 1.5% dari total SHAP magnitude (`MIN_CONTRIBUTION_PCT`) |
| Endpoint yang mendukung | **Hanya** `/predict/existing` — lihat Bagian 4 |

Setiap faktor risiko direpresentasikan sebagai objek `ReasonItem` dengan 5 field:

| Field | Tipe | Contoh Isi |
|---|---|---|
| `feature` | `str` | `"EXT_SOURCE_2"` |
| `label` | `str` | `"Skor riwayat kredit dari lembaga eksternal 2"` |
| `value_display` | `str` | `"0.72 (skala 0-1)"` |
| `direction` | `Literal` | `"menaikkan_risiko"` / `"menurunkan_risiko"` |
| `contribution_pct` | `float` | `18.4` |

## 3. Contoh Kasus Nyata

Berikut ilustrasi bagaimana `ReasonItem` mentah diterjemahkan jadi kalimat yang mudah
dipahami reviewer kredit (bukan cuma angka teknis):

**Fitur: `EXT_SOURCE_2` (skor 0.0, hasil imputasi/placeholder untuk data hilang)**
> "Tidak ada riwayat skor dari lembaga eksternal 2 — nasabah baru belum memiliki data
> dari lembaga ini."

**Fitur: `PREV_REJECTION_RATIO` (rasio 0.4)**
> "40% dari pengajuan kredit nasabah sebelumnya ditolak — riwayat ini meningkatkan
> risiko."

**Fitur: `CC_UTILIZATION_RATIO` (rasio 1.15)**
> "Saldo kartu kredit nasabah pernah melebihi limit sebesar 15% dalam 6 bulan
> terakhir."

**Fitur: `TOURISM_RATIO` (rasio 0.05)**
> "Hanya 5% pengajuan sebelumnya untuk kategori barang pariwisata — kategori ini
> cenderung menurunkan risiko."

Kalimat-kalimat ini dihasilkan oleh fungsi formatter khusus per fitur (bukan template
generik), supaya reviewer non-teknis tetap paham konteks bisnisnya, bukan cuma nama
kolom mentah.

## 4. Keterbatasan yang Perlu Diketahui

Bagian ini sengaja ditulis eksplisit — kejujuran soal batasan sistem adalah bagian
dari praktik *responsible ML*, bukan kelemahan yang harus disembunyikan.

- **SHAP reasons hanya tersedia di Mode Existing (`/predict/existing`).** Mode Baru
  (`/predict/new`) tidak mengembalikan `reasons` sama sekali (selalu `null`), karena
  nasabah baru tidak punya histori fitur agregasi yang jadi basis sebagian besar top
  predictor. Keputusan Mode Baru selalu dikunci `REVIEW/REJECT` terlepas dari skor
  probabilitas, jadi penjelasan SHAP granular untuk mode ini tidak relevan — yang
  relevan adalah disclaimer out-of-distribution (lihat dokumentasi utama di README).
- **Daftar faktor risiko yang punya kalimat naratif khusus masih tetap (fixed),**
  bukan otomatis mengikuti fitur apa pun yang muncul di top-5 SHAP. Saat ini hanya
  4 fitur (`EXT_SOURCE_2`, `PREV_REJECTION_RATIO`, `CC_UTILIZATION_RATIO`,
  `TOURISM_RATIO`) punya formatter khusus. Kalau top-5 SHAP untuk seorang nasabah
  memuat fitur di luar 4 ini, sistem akan menampilkan nama fitur teknis tanpa
  kalimat naratif yang dipoles.
- **Filter minimum kontribusi 1.5% berarti fitur dengan pengaruh sangat kecil tidak
  ditampilkan**, walau secara teknis tetap berkontribusi terhadap skor akhir.