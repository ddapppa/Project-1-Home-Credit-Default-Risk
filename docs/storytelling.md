# Storytelling: Membangun Sistem Prediksi Risiko Kredit yang Tahu Batasannya Sendiri

## 1. Problem — Kenapa Project Ini Dibuat

Setiap tim credit risk punya masalah yang sama: dua jenis kesalahan yang sama-sama mahal, tapi mahalnya beda jauh.

Kalau model salah menyetujui pemohon yang ternyata gagal bayar (False Negative), perusahaan kehilangan pokok pinjaman plus bunga yang tidak pernah kembali — kerugian riil, langsung terasa di laporan keuangan. Kalau model salah menolak pemohon yang sebenarnya layak (False Positive), kerugiannya lebih "diam" — cuma kehilangan potensi bunga dari satu nasabah yang pindah ke kompetitor. Kedua kesalahan itu nyata, tapi tidak setara.

Masalahnya, hampir semua tutorial machine learning mengajarkan untuk berhenti begitu model punya akurasi atau AUC yang bagus. Tidak ada yang mengajarkan: kalau dua jenis kesalahan itu punya bobot berbeda, bagaimana cara menerjemahkan itu ke dalam angka yang benar-benar dipakai sistem untuk mengambil keputusan?

Itu pertanyaan yang saya coba jawab lewat project ini — bukan cuma "bisa memprediksi risiko default", tapi "bisa mengambil keputusan yang masuk akal secara bisnis, dan tahu kapan harus mengaku tidak yakin".

## 2. Pendekatan — Bagaimana Saya Menyelesaikannya

Saya mulai dari dataset resmi kompetisi Kaggle Home Credit Default Risk — 307.511 baris data aplikasi pinjaman, digabung dengan riwayat kredit eksternal (bureau), riwayat aplikasi sebelumnya, dan riwayat pembayaran cicilan.

Prosesnya berjalan iteratif, bukan sekali jadi. Feature engineering saya lewati tiga versi (v1 sampai v3) — setiap iterasi menambah kedalaman fitur agregasi dari tabel-tabel tambahan (bureau, previous_application, installments_payments), sambil terus mengecek apakah fitur baru benar-benar menambah sinyal atau cuma menambah noise lewat Information Value (IV) ranking dan korelasi.

Untuk pemilihan model, saya bandingkan 14 kandidat — dari Logistic Regression sederhana sampai gradient boosting (LightGBM, XGBoost, CatBoost) dan ensemble stacking. CatBoost v3 akhirnya jadi model final, bukan karena "yang paling canggih", tapi karena punya AUC-ROC tertinggi (0.7862) sekaligus native handling untuk fitur kategorikal tanpa perlu encoding manual berlebihan — cocok dengan karakteristik dataset yang punya banyak kolom kategorikal (tipe pekerjaan, tipe kontrak, dll).

Tapi bagian paling penting bukan di sini. Model yang bagus itu titik awal, bukan titik akhir. Bagian yang menurut saya paling menunjukkan cara berpikir saya sebagai data scientist ada di dua keputusan berikut.

## 3. Keputusan Sulit — Bagian yang Paling Menunjukkan Cara Berpikir

### 3.1 Kenapa Threshold Bukan 0.5, Tapi 0.6669 — dan Kenapa Itu Bukan Angka Sembarangan

Default di hampir semua tutorial klasifikasi biner: kalau probabilitas > 0.5, prediksi positif. Saya sengaja tidak pakai itu.

Alasannya sederhana: threshold 0.5 mengasumsikan bahwa salah menyetujui pemohon gagal bayar dan salah menolak pemohon layak itu **sama mahalnya**. Di dunia kredit, itu asumsi yang salah. Kehilangan pokok pinjaman dari nasabah yang default jauh lebih merugikan dibanding kehilangan potensi bunga dari nasabah layak yang ditolak.

Saya terjemahkan itu jadi rasio biaya eksplisit: **False Negative : False Positive = 5:1**. Artinya, menyetujui pemohon yang ternyata gagal bayar dianggap 5 kali lebih merugikan dibanding menolak pemohon yang sebenarnya layak. Dari rasio ini, saya hitung threshold optimal murni dari **Out-of-Fold prediction** (hasil cross-validation, bukan dari holdout set) — hasilnya 0.6669.

Kenapa harus dari Out-of-Fold, bukan holdout? Karena kalau threshold dioptimasi langsung dari holdout set, saya sebenarnya "mengintip" data yang seharusnya jadi ujian akhir yang independen. Itu bentuk data leakage yang halus tapi nyata — model jadi terlihat lebih bagus di atas kertas, padahal performanya di dunia nyata (data yang belum pernah dilihat) bisa jauh lebih buruk. Threshold 0.6669 ini murni hasil dari data training, lalu diuji "buta" ke holdout set yang di-freeze sejak awal project — dan hasilnya tetap konsisten (Precision 27.9%, Recall 44.6%).

**Ini bagian yang saya anggap paling penting dari seluruh project**, karena ini bukan soal "model bisa prediksi", tapi soal menerjemahkan asumsi bisnis (biaya kesalahan yang tidak simetris) menjadi keputusan matematis yang bisa dipertanggungjawabkan — bahasa yang dimengerti hiring manager bisnis, bukan cuma sesama data scientist.

### 3.2 Kenapa Mode Baru Tidak Pernah Auto-Approve — Model yang Tahu Batasannya Sendiri

Saya membangun dua mode di backend: **Mode Existing** (lookup nasabah dari data holdout, dengan seluruh 739 fitur tersedia — kondisi ideal) dan **Mode Baru** (form manual untuk nasabah tanpa histori kredit sama sekali).

Saat menguji Mode Baru, saya cek satu hal yang jarang dicek kandidat lain: apakah kombinasi data yang dimasukkan nasabah baru ini **pernah muncul** di data training? Hasilnya mengejutkan — **0 dari 307.511 baris training** punya kombinasi identik dengan skenario "tanpa histori kredit, tanpa skor eksternal, murni data form manual".

Ini artinya, setiap prediksi di Mode Baru itu **out-of-distribution** — model diminta menjawab pertanyaan yang tidak pernah dia pelajari jawabannya. Model gradient boosting seperti CatBoost tetap akan mengeluarkan angka probabilitas berapa pun inputnya (itu sifat matematisnya), tapi angka itu tidak punya dasar empiris apa pun untuk skenario ini.

Godaan paling umum di sini adalah tetap menampilkan angka probabilitas itu seolah-olah valid, karena "kan modelnya tetap jalan". Saya memilih tidak melakukan itu. Sebagai gantinya, saya kunci keputusan Mode Baru selalu ke status **"Perlu Verifikasi Manual"**, terlepas dari nilai probabilitas yang keluar — tidak pernah auto-approve, tidak peduli seberapa "aman" angkanya terlihat.

Keputusan ini terasa kontra-intuitif dari sisi "showcase kemampuan model", karena secara teknis lebih mudah membiarkan model tetap memberi keputusan otomatis. Tapi dari sisi tanggung jawab produksi, ini keputusan yang lebih matang: **mengakui batas model sendiri lebih penting daripada terlihat lebih canggih**. Sistem kredit yang percaya diri berlebihan terhadap data yang tidak pernah dia lihat justru lebih berbahaya daripada sistem yang jujur bilang "saya tidak yakin, butuh manusia untuk cek ini".

### 3.3 Fairness — Kenapa `CODE_GENDER` Tidak Dipakai sebagai Fitur

Satu keputusan kecil yang jarang disorot tapi sengaja saya ambil: kolom `CODE_GENDER` (jenis kelamin pemohon) tidak saya masukkan sebagai fitur prediktif, meskipun secara statistik kolom ini punya sedikit sinyal terhadap target.

Alasannya bukan soal akurasi, tapi soal **fair lending** — praktik pemberian kredit yang tidak boleh mendiskriminasi berdasarkan atribut yang dilindungi secara hukum di banyak negara (gender, ras, dll). Model yang secara diam-diam belajar pola dari atribut seperti ini bisa menghasilkan keputusan yang secara teknis akurat tapi secara etis dan legal bermasalah. Untuk project yang mensimulasikan sistem kredit dunia nyata, saya anggap ini bukan detail teknis kecil — ini bagian dari desain sistem yang bertanggung jawab.

## 4. Hasil — Apa yang Tercapai

Model final (CatBoost v3) diuji pada holdout set sebanyak 61.503 baris yang di-freeze sejak awal project (stratified 80/20 split, tidak pernah disentuh selama proses tuning):

| Metrik | Nilai |
|---|---|
| ROC-AUC | 0.7862 |
| PR-AUC | 0.2838 |
| Precision | 27.9% |
| Recall | 44.6% |
| F1-Score | 0.343 |

Di luar angka-angka ini, tiga hal yang saya anggap paling bernilai dari seluruh project:

1. **Threshold keputusan yang berdasar rasio biaya bisnis nyata**, bukan default arbitrer 0.5, dan divalidasi tanpa data leakage
2. **Sistem yang secara aktif mendeteksi dan menolak memprediksi skenario di luar jangkauan datanya sendiri**, alih-alih memaksa memberi jawaban
3. **Pertimbangan fairness dijadikan bagian dari desain fitur**, bukan renungan belakangan

Seluruh proses ini didukung backend FastAPI yang sudah diuji end-to-end secara lokal, dengan explainability lewat SHAP untuk setiap prediksi di Mode Existing — supaya setiap keputusan model bisa dijelaskan ke tim underwriting non-teknis, bukan cuma jadi angka misterius dari kotak hitam.

## 5. Kalau Diulang Lagi

Kalau saya bangun ulang project ini dari awal, satu hal yang ingin saya lakukan lebih dini: menguji deteksi out-of-distribution di Mode Baru **sebelum** membangun seluruh pipeline modeling, bukan setelah model final selesai. Menemukan masalah ini di akhir berarti saya harus mendesain ulang logika keputusan aplikasi setelah semua sudah "jalan" — pelajaran yang saya bawa untuk project berikutnya: validasi asumsi tentang jangkauan data sedini mungkin, bukan setelah model dianggap selesai.