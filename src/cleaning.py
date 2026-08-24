# 1. Konfigurasi & Fungsi Load/Subset Data

"""Bagian ini mendefinisikan daftar kolom yang dipakai berulang di seluruh fungsi (grup housing, kolom bureau, dll.), lalu dua fungsi awal: membaca daftar 91 fitur dari selected_features_eda.csv, dan memfilter data mentah 122 kolom supaya hanya menyisakan 91 
fitur + ID + TARGET. Ini wajib jadi langkah pertama karena kamu belum punya data 91 fitur itu, cuma data mentah."""

import numpy as np
import pandas as pd


RARE_CATEGORY_THRESHOLD = 50
DAYS_EMPLOYED_ANOMALY_VALUE = 365243

HOUSING_NUMERIC_COLS = [
    "FLOORSMAX_AVG", "FLOORSMAX_MEDI", "FLOORSMAX_MODE",
    "TOTALAREA_MODE",
    "LIVINGAREA_AVG", "LIVINGAREA_MEDI", "LIVINGAREA_MODE",
    "APARTMENTS_AVG", "APARTMENTS_MEDI", "APARTMENTS_MODE",
    "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BEGINEXPLUATATION_MODE",
    "ENTRANCES_AVG", "ENTRANCES_MEDI", "ENTRANCES_MODE",
]
HOUSING_CATEGORICAL_COLS = ["WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE", "HOUSETYPE_MODE"]
HOUSING_INDICATOR_COL = "TOTALAREA_MODE"

AMT_REQ_CREDIT_BUREAU_COLS = [
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
]

FINANCIAL_LOG_COLS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]

RARE_CATEGORY_COLS = ["NAME_INCOME_TYPE", "ORGANIZATION_TYPE", "OCCUPATION_TYPE"]

NEAR_ZERO_VARIANCE_COLS = [
    "FLAG_MOBIL", "FLAG_CONT_MOBILE",
    "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_9",
    "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
]


def load_selected_features(csv_path):
    """Baca daftar 91 fitur hasil seleksi Bagian 5 EDA dari selected_features_eda.csv."""
    selected_df = pd.read_csv(csv_path)
    return selected_df["feature"].tolist()


def subset_selected_features(df, selected_features, id_col="SK_ID_CURR", target_col="TARGET"):
    """
    Filter dataframe mentah (122 kolom) supaya hanya menyisakan 91 fitur hasil
    seleksi EDA + ID + TARGET. Langkah pertama sebelum semua proses cleaning
    lain, karena data yang kamu punya masih data mentah 122 kolom.
    """
    cols_to_keep = [id_col] + selected_features
    if target_col in df.columns:
        cols_to_keep = [id_col, target_col] + selected_features
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    return df[cols_to_keep].copy()

# 2. Anomali DAYS_EMPLOYED

"""Replace sentinel 365243 jadi NaN, buat flag khusus untuk 22 baris anomali yang bukan Pensioner (temuan cross-check EDA),
lalu impute sisa NaN dengan median — karena model non-tree butuh nilai numerik utuh, tidak bisa dibiarkan NaN."""

def fix_days_employed_anomaly(df):
    """
    Replace sentinel anomaly 365243 pada DAYS_EMPLOYED menjadi NaN, lalu buat
    flag DAYS_EMPLOYED_ANOMALY untuk baris anomaly yang BUKAN Pensioner (22
    baris dari EDA, kategori Unemployed) supaya sinyal itu tidak hilang saat
    DAYS_EMPLOYED diimputasi median di langkah berikutnya.
    """
    df = df.copy()
    anomaly_mask = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY_VALUE

    if "NAME_INCOME_TYPE" in df.columns:
        non_pensioner_anomaly = anomaly_mask & (df["NAME_INCOME_TYPE"] != "Pensioner")
    else:
        non_pensioner_anomaly = anomaly_mask

    df["DAYS_EMPLOYED_ANOMALY"] = non_pensioner_anomaly.astype(int)
    df.loc[anomaly_mask, "DAYS_EMPLOYED"] = np.nan
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].fillna(df["DAYS_EMPLOYED"].median())
    return df


# 3. Hidden Missing Value & Imputasi Kategori Kecil
"""Replace placeholder "XNA" (bukan "Other", sesuai koreksi dari diskusi kita) jadi NaN di tiga kolom, 
lalu imputasi CODE_GENDER (4 baris) dan NAME_FAMILY_STATUS (2 baris) dengan modus karena jumlahnya sangat kecil."""

def fix_hidden_missing_placeholders(df):
    """
    Replace placeholder string yang merepresentasikan missing value tersembunyi
    ("XNA") menjadi NaN pada ORGANIZATION_TYPE, CODE_GENDER, dan NAME_FAMILY_STATUS.
    PENTING: kategori "Other" pada ORGANIZATION_TYPE TIDAK disentuh karena itu
    kategori valid (catch-all), bukan missing value tersembunyi.
    """
    df = df.copy()
    placeholder_map = {
        "ORGANIZATION_TYPE": "XNA",
        "CODE_GENDER": "XNA",
        "NAME_FAMILY_STATUS": "Unknown",
    }
    for col, placeholder in placeholder_map.items():
        if col in df.columns:
            df[col] = df[col].replace(placeholder, np.nan)
    return df


def impute_gender_and_family_status(df):
    """
    Imputasi CODE_GENDER (4 baris XNA) dan NAME_FAMILY_STATUS (2 baris Unknown)
    dengan modus. Jumlahnya sangat kecil (<0,01%) sehingga tidak perlu kategori
    "Missing" terpisah, cukup diisi nilai paling umum.
    """
    df = df.copy()
    for col in ["CODE_GENDER", "NAME_FAMILY_STATUS"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val)
    return df


#4. Imputasi Grup Fitur Housing (Missing Terstruktur)
"""Berdasarkan temuan 3.c di EDA (korelasi missing antar kolom housing sangat tinggi), buat satu flag HAS_HOUSING_INFO, 
lalu impute median untuk kolom numerik dan kategori "Missing" untuk kolom kategorikal — bukan imputasi independen per kolom."""

def impute_housing_group(df):
    """
    Imputasi grup fitur housing yang missing 47-51% dan terbukti missing
    BERSAMAAN (temuan 3.c EDA). Buat satu flag HAS_HOUSING_INFO, lalu impute
    median untuk kolom numerik dan kategori "Missing" untuk kolom kategorikal.
    """
    df = df.copy()
    existing_indicator = HOUSING_INDICATOR_COL if HOUSING_INDICATOR_COL in df.columns else None
    if existing_indicator:
        df["HAS_HOUSING_INFO"] = df[existing_indicator].notnull().astype(int)
    else:
        df["HAS_HOUSING_INFO"] = 1

    for col in HOUSING_NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in HOUSING_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")

    return df


#5. Imputasi EXT_SOURCE_1 & EXT_SOURCE_3
"""Median + flag _MISSING per kolom, karena bin "Missing" di perhitungan IV Bagian 4 EDA terbukti informatif, bukan sekadar noise acak."""

def impute_ext_source(df):
    """
    Imputasi EXT_SOURCE_1 (missing 56%) dan EXT_SOURCE_3 (missing 20%) dengan
    median, plus flag _MISSING per kolom, karena bin "Missing" di perhitungan
    IV terbukti informatif.
    """
    df = df.copy()
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_3"]:
        if col in df.columns:
            df[f"{col}_MISSING"] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(df[col].median())
    return df


#6. Imputasi AMT_REQ_CREDIT_BUREAU_*
"""Impute dengan 0, asumsi tidak ada rekaman inquiry = tidak ada inquiry bureau kredit."""

def impute_credit_bureau_inquiries(df):
    """
    Imputasi AMT_REQ_CREDIT_BUREAU_* (missing 13,5% di semua kolom) dengan 0,
    berdasarkan asumsi tidak ada rekaman inquiry = tidak ada inquiry bureau
    kredit, bukan data hilang acak.
    """
    df = df.copy()
    for col in AMT_REQ_CREDIT_BUREAU_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df

# 7. Imputasi OCCUPATION_TYPE
"""Fungsi ini baru dipanggil setelah verifikasi cross-tab di notebook mengonfirmasi keputusan nama kategori. Parameter replacement_category sengaja dibuat fleksibel, 
bukan hardcode "Not_Working", supaya kamu bisa sesuaikan setelah lihat hasil verifikasinya nanti."""

def impute_occupation_type(df, replacement_category="Not_Working"):
    """
    Imputasi OCCUPATION_TYPE (missing 31,35%) dengan kategori replacement_category.
    PENTING: fungsi ini baru dipanggil SETELAH verifikasi cross-tab di notebook
    02_Cleaning.ipynb mengonfirmasi bahwa missing-nya memang berasal dari klien
    yang tidak bekerja (Pensioner/Unemployed/dst), bukan missing acak.
    """
    df = df.copy()
    if "OCCUPATION_TYPE" in df.columns:
        df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna(replacement_category)
    return df

# 8. Log-Transform (Kolom Baru, Bukan Overwrite)
"""Buat kolom LOG_* untuk 4 fitur finansial sesuai outlier_treatment_decision.csv. Kolom asli tetap disimpan. 
MT_ANNUITY (12 missing) dan AMT_GOODS_PRICE (278 missing) diimputasi median dulu supaya transform-nya tidak menghasilkan NaN."""

def create_log_features(df):
    """
    Buat kolom baru LOG_* (log1p) untuk 4 fitur finansial sesuai rekomendasi
    outlier_treatment_decision.csv. Kolom asli TETAP disimpan (tidak di-overwrite)
    supaya model tree-based bisa memakai versi asli, sedangkan model linear/
    KNN/neural net memakai versi log. Imputasi median dilakukan dulu untuk
    AMT_ANNUITY (12 missing) dan AMT_GOODS_PRICE (278 missing) sebelum transform.
    """
    df = df.copy()
    for col in ["AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in FINANCIAL_LOG_COLS:
        if col in df.columns:
            df[f"LOG_{col}"] = np.log1p(df[col])
    return df

# 9. Rare Category Consolidation
"""Gabungkan kategori dengan n < 50 jadi "Rare" untuk tiga kolom kategorikal dengan kategori sangat jarang 
(NAME_INCOME_TYPE, ORGANIZATION_TYPE, OCCUPATION_TYPE) — sesuai kesepakatan diskusi kita soal "coarse classing"""

def consolidate_rare_categories(df, cols=RARE_CATEGORY_COLS, threshold=RARE_CATEGORY_THRESHOLD):
    """
    Gabungkan kategori dengan jumlah sampel < threshold jadi kategori "Rare".
    Prinsipnya sama dengan smoothing epsilon di perhitungan IV - kategori
    dengan sampel sangat kecil tidak boleh menghasilkan estimasi yang tidak
    stabil (contoh: NAME_INCOME_TYPE "Maternity leave" cuma 5 baris).
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            rare_categories = counts[counts < threshold].index
            df[col] = df[col].apply(lambda x: "Rare" if x in rare_categories else x)
    return df

# 10. Tipe Data Fitur Near-Zero Variance
"""Rapikan tipe data 18 fitur FLAG_DOCUMENT_*/FLAG_MOBIL/FLAG_CONT_MOBILE jadi int8. Tidak dibuang di sini 
keputusan buang menunggu VarianceThreshold di 03_feature_engineering.ipynb, sesuai kesepakatan kita sebelumnya."""

def ensure_near_zero_variance_dtype(df, cols=NEAR_ZERO_VARIANCE_COLS):
    """
    Pastikan tipe data 18 fitur near-zero-variance jadi int8 untuk efisiensi
    memori. Fitur ini TIDAK dibuang di tahap cleaning - keputusan buang
    menunggu VarianceThreshold di 03_feature_engineering.ipynb.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("int8")
    return df


# 11. Safety Net & Validasi Akhir (Tambahan Penting yang Belum Ada di EDA)
"""Ini yang saya tambahkan di luar hasil analisis EDA, tapi krusial: setelah semua imputasi spesifik, cek dan tangani sisa NaN apapun yang mungkin terlewat (misalnya DAYS_LAST_PHONE_CHANGE yang punya 1 baris missing tapi tidak masuk ke kategori spesifik manapun di atas), lalu validasi akhir yang wajib memastikan nol NaN 
karena model non-tree-based yang jadi target project ini tidak bisa mentolerir missing value sama sekali."""

def impute_remaining_missing(df, exclude_cols=("SK_ID_CURR", "TARGET")):
    """
    Safety net: setelah semua imputasi spesifik di atas, tangani SISA missing
    value apapun yang belum tercakup (misal DAYS_LAST_PHONE_CHANGE 1 baris).
    WAJIB dijalankan karena model non-tree-based (Logistic Regression, KNN,
    SVM, dll.) tidak bisa menangani NaN sama sekali, tidak seperti LightGBM/
    XGBoost yang punya penanganan missing built-in.
    """
    df = df.copy()
    cols_to_check = [c for c in df.columns if c not in exclude_cols]
    for col in cols_to_check:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
    return df


def validate_no_missing(df, exclude_cols=("SK_ID_CURR", "TARGET")):
    """
    Validasi akhir: pastikan TIDAK ADA missing value tersisa di seluruh kolom
    (kecuali ID/TARGET). Raise error kalau masih ada, supaya masalah terdeteksi
    di sini, bukan menyebabkan error tersembunyi saat training model nanti.
    """
    cols_to_check = [c for c in df.columns if c not in exclude_cols]
    remaining_missing = df[cols_to_check].isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    if len(remaining_missing) > 0:
        raise ValueError(f"Masih ada missing value di kolom: {remaining_missing.to_dict()}")
    print("Validasi berhasil: tidak ada missing value tersisa.")
    return True


def impute_organization_type(df, replacement_category="Not_Working"):
    """
    Imputasi ORGANIZATION_TYPE (55.374 baris NaN setelah fix_hidden_missing_placeholders,
    berasal dari placeholder "XNA") dengan replacement_category. Akar sebabnya SAMA
    dengan OCCUPATION_TYPE - klien Pensioner/Unemployed tidak punya organisasi kerja.
    PENTING: fungsi ini harus dipanggil SEBELUM consolidate_rare_categories(), karena
    NaN dalam jumlah besar tidak akan dianggap "rare" oleh threshold 50, dan akan
    jatuh ke modus-fill di impute_remaining_missing() kalau tidak ditangani di sini.
    """
    df = df.copy()
    if "ORGANIZATION_TYPE" in df.columns:
        df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].fillna(replacement_category)
    return df


# 11. Duplicate Check
"""Fungsi ini mengecek dua jenis duplikasi: duplikat baris penuh
(semua kolom identik — otomatis dibuang karena tidak menambah informasi apapun) dan duplikat ID (SK_ID_CURR sama tapi isi baris berbeda — ini justru indikasi masalah data yang tidak boleh dibuang otomatis, harus diselidiki manual karena bisa jadi kesalahan penggabungan data)."""


def check_duplicates(df, id_col="SK_ID_CURR", drop_full_duplicates=True):
    """
    Cek duplikasi baris (baik duplikat penuh maupun duplikat ID) sebagai
    validasi kualitas data dasar sebelum diproses lebih lanjut. Baris
    duplikat penuh (semua kolom identik) otomatis dibuang karena tidak
    memberi informasi tambahan; duplikat ID (SK_ID_CURR sama tapi isi
    baris berbeda) TIDAK otomatis dibuang karena itu indikasi masalah
    data yang perlu diselidiki manual.
    """
    df = df.copy()
    n_full_duplicates = df.duplicated().sum()
    n_id_duplicates = df[id_col].duplicated().sum() if id_col in df.columns else 0

    report: dict[str, int | str] = {
        "n_full_duplicate_rows": int(n_full_duplicates),
        "n_id_duplicates": int(n_id_duplicates),
    }

    if n_full_duplicates > 0 and drop_full_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        report["action"] = f"Dibuang {n_full_duplicates} baris duplikat penuh."
    elif n_id_duplicates > 0:
        report["action"] = (
            f"PERHATIAN: {n_id_duplicates} SK_ID_CURR duplikat dengan isi baris"
            "berbeda - perlu investigasi manual, TIDAK dibuang otomatis."
        )
    else:
        report["action"] = "Tidak ada duplikasi ditemukan."

    return df, report

# Exact-Duplicate-Column Check
"""Fungsi ini mengecek apakah ada dua kolom atau lebih yang isinya 100% identik (byte-for-byte), bukan cuma berkorelasi tinggi. Ini murni soal redundansi absolut — berbeda dengan pengecekan korelasi statistik (Pearson/Spearman/Cramér's V) yang tetap jadi tanggung jawab 03_feature_engineering.ipynb. Fungsi ini hanya melaporkan, tidak otomatis membuang kolom, 
supaya keputusan pembuangan tetap dikonfirmasi manual di notebook."""

def check_exact_duplicate_columns(df, exclude_cols=("SK_ID_CURR", "TARGET")):
    """
    Cek apakah ada dua kolom atau lebih yang isinya 100% identik (byte-for-byte),
    bukan cuma berkorelasi tinggi. Ini murni soal kualitas data (redundansi
    absolut) - berbeda dengan pengecekan korelasi statistik yang tetap jadi
    tanggung jawab 03_feature_engineering.ipynb. Fungsi ini hanya MELAPORKAN,
    tidak otomatis membuang kolom - keputusan pembuangan tetap perlu
    dikonfirmasi manual di notebook.
    """
    cols_to_check = [c for c in df.columns if c not in exclude_cols]
    duplicate_groups = []
    checked = set()

    for i, col_a in enumerate(cols_to_check):
        if col_a in checked:
            continue
        group = [col_a]
        for col_b in cols_to_check[i + 1:]:
            if col_b in checked:
                continue
            if df[col_a].equals(df[col_b]):
                group.append(col_b)
                checked.add(col_b)
        if len(group) > 1:
            duplicate_groups.append(group)
            checked.add(col_a)

    return duplicate_groups

# Validasi Konsistensi Tipe Data & Kategori (Bagian 11, Bersama validate_no_missing)
"""Fungsi ini jadi validasi akhir kedua (selain validate_no_missing) — memastikan FLAG_* cuma berisi {0, 1}, CODE_GENDER cuma berisi {"M", "F"} setelah fix_hidden_missing_placeholders, dan kolom DAYS_* tetap bernilai <=0 setelah semua imputasi. Kalau ada pelanggaran, 
langsung raise ValueError — supaya kesalahan tipe data ketahuan di sini, bukan menyebabkan bug tersembunyi saat modeling nanti."""

DAYS_COLS_TO_CHECK = [
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
]

FLAG_YN_COLS = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]

def validate_dtypes_and_categories(df):
    """
    Validasi konsistensi tipe data dan kategori setelah seluruh proses
    cleaning selesai.

    Aturan validasi:
    - Kolom FLAG_* numerik harus hanya berisi 0 atau 1.
    - FLAG_OWN_CAR dan FLAG_OWN_REALTY harus berisi "Y" atau "N".
    - CODE_GENDER harus berisi "M" atau "F".
    - Kolom DAYS_* harus bernilai <= 0 karena merepresentasikan hari
      sebelum tanggal pengajuan aplikasi.
    """
    issues = []

    flag_cols = [
        col
        for col in df.columns
        if col.startswith("FLAG_") and col not in FLAG_YN_COLS
    ]

    for col in flag_cols:
        invalid_values = set(df[col].dropna().unique()) - {0, 1}
        if invalid_values:
            issues.append(f"{col} punya nilai selain 0/1: {invalid_values}")

    for col in FLAG_YN_COLS:
        if col in df.columns:
            invalid_values = set(df[col].dropna().unique()) - {"Y", "N"}
            if invalid_values:
                issues.append(f"{col} punya nilai selain Y/N: {invalid_values}")

    if "CODE_GENDER" in df.columns:
        invalid_gender = set(df["CODE_GENDER"].dropna().unique()) - {"M", "F"}
        if invalid_gender:
            issues.append(
                f"CODE_GENDER punya nilai selain M/F: {invalid_gender}"
            )

    for col in DAYS_COLS_TO_CHECK:
        if col in df.columns and (df[col] > 0).any():
            n_positive = (df[col] > 0).sum()
            issues.append(
                f"{col} punya {n_positive} baris bernilai positif "
                "(seharusnya <= 0)"
            )

    if issues:
        raise ValueError(
            "Ditemukan inkonsistensi tipe data/kategori:\n"
            + "\n".join(issues)
        )

    print("Validasi berhasil: semua tipe data dan kategori konsisten.")
    return True