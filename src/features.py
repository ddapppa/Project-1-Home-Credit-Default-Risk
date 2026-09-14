# 1. Konfigurasi & Fungsi Load Cleaned Dataset
"""Bagian ini mendefinisikan fungsi untuk membaca dataset hasil Tahap 3 Cleaning
(application_train_clean.csv), plus konstanta threshold redundancy yang akan
dipakai di seluruh fungsi korelasi. Ini wajib jadi langkah pertama karena
03_feature_engineering.ipynb bekerja di atas data yang SUDAH bersih, bukan
data mentah lagi."""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.feature_selection import VarianceThreshold

from cleaning import HOUSING_NUMERIC_COLS, NEAR_ZERO_VARIANCE_COLS


REDUNDANCY_THRESHOLD = 0.9
HOUSING_VARIANT_SUFFIXES = ("_AVG", "_MEDI", "_MODE")
HOUSING_PREFERRED_SUFFIX = "_MEDI"


def load_cleaned_dataset(csv_path):
    """
    Baca dataset hasil Tahap 3 Cleaning (application_train_clean.csv).

    Fungsi ini terpisah dari load_selected_features() di cleaning.py karena
    tujuannya berbeda: di sini kita load data yang SUDAH melalui seluruh
    pipeline cleaning (93 fitur awal + 8 kolom baru = 101 kolom), bukan
    daftar 91 kandidat fitur mentah dari hasil EDA.
    """
    df = pd.read_csv(csv_path)
    return df


# 2. Analisis Korelasi Grup Housing (_AVG / _MEDI / _MODE)
"""Grup fitur housing punya 3 varian (_AVG, _MEDI, _MODE) untuk hampir setiap
fitur dasar yang sama (misal APARTMENTS_AVG, APARTMENTS_MEDI, APARTMENTS_MODE).
Sebelum memutuskan representasi mana yang dipertahankan, kita perlu bukti
korelasi intra-grup (seberapa mirip ketiga varian) DAN korelasi masing-masing
varian terhadap TARGET (representasi mana yang paling informatif untuk model).
Fungsi build_housing_groups() dibuat dinamis dari HOUSING_NUMERIC_COLS supaya
tidak hardcode ulang nama kolom yang sudah didefinisikan di cleaning.py."""


def build_housing_groups(cols=HOUSING_NUMERIC_COLS):
    """
    Kelompokkan kolom housing berdasarkan nama fitur dasarnya (base name),
    dengan memisahkan suffix (akhiran nama kolom) _AVG/_MEDI/_MODE.

    Contoh: APARTMENTS_AVG, APARTMENTS_MEDI, APARTMENTS_MODE
            -> dikelompokkan jadi base name "APARTMENTS".

    Kolom yang HANYA punya satu varian (contoh: TOTALAREA_MODE, tidak ada
    TOTALAREA_AVG/TOTALAREA_MEDI) otomatis dikeluarkan dari hasil karena
    tidak relevan untuk analisis redundancy intra-grup.
    """
    groups = {}
    for col in cols:
        for suffix in HOUSING_VARIANT_SUFFIXES:
            if col.endswith(suffix):
                base_name = col[: -len(suffix)]
                groups.setdefault(base_name, []).append(col)
                break

    return {base: variant_cols for base, variant_cols in groups.items() if len(variant_cols) > 1}


def compute_housing_variant_correlation(df, groups=None):
    """
    Hitung korelasi Pearson antar varian (_AVG vs _MEDI vs _MODE) untuk setiap
    fitur housing dasar. Mengembalikan dict {base_name: correlation_matrix}.

    Korelasi tinggi (mendekati REDUNDANCY_THRESHOLD atau lebih) jadi bukti
    bahwa ketiga varian memang sangat mirip, sehingga aman untuk memilih
    HANYA SATU representasi tanpa kehilangan banyak informasi.
    """
    if groups is None:
        groups = build_housing_groups()

    correlation_results = {}
    for base_name, variant_cols in groups.items():
        existing_cols = [c for c in variant_cols if c in df.columns]
        if len(existing_cols) > 1:
            correlation_results[base_name] = df[existing_cols].corr(method="pearson").round(4)

    return correlation_results


def compute_variant_target_correlation(df, groups=None, target_col="TARGET"):
    """
    Hitung korelasi Pearson setiap varian housing (_AVG/_MEDI/_MODE) terhadap
    TARGET. Dipakai sebagai bukti pendukung untuk memilih representasi mana
    yang paling informatif terhadap default risk, BUKAN sekadar memilih
    secara acak/asumsi (misal selalu pakai _AVG tanpa cek data).

    Mengembalikan DataFrame dengan kolom: base_name, variant, corr_with_target,
    diurutkan dari korelasi absolut tertinggi ke terendah per grup.
    """
    if groups is None:
        groups = build_housing_groups()

    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dataframe.")

    records = []
    for base_name, variant_cols in groups.items():
        for col in variant_cols:
            if col in df.columns:
                corr_value = df[col].corr(df[target_col], method="pearson")
                records.append({
                    "base_name": base_name,
                    "variant": col,
                    "corr_with_target": round(corr_value, 4),
                })

    result_df = pd.DataFrame(records)
    result_df["abs_corr"] = result_df["corr_with_target"].abs()
    result_df = result_df.sort_values(
        ["base_name", "abs_corr"], ascending=[True, False]
    ).drop(columns="abs_corr").reset_index(drop=True)

    return result_df


# 3. Seleksi Representasi Fitur Housing Final
"""Terapkan keputusan yang diambil setelah analisis Langkah 2: seluruh grup
housing menunjukkan korelasi intra-grup > 0.96 (redundan) dan selisih korelasi
ke TARGET antar varian tidak signifikan (di desimal ketiga-keempat). Karena
itu dipilih SATU representasi konsisten untuk semua grup, yaitu _MEDI, dengan
alasan median lebih robust terhadap outlier dibanding mean (_AVG) - bukan
dipilih per grup berdasarkan angka korelasi tertinggi yang berpotensi noise."""


def select_housing_representative(df, groups=None, preferred_suffix=HOUSING_PREFERRED_SUFFIX):
    """
    Buang kolom housing redundan, pertahankan hanya representasi _MEDI untuk
    setiap fitur dasar. TOTALAREA_MODE tidak terdampak karena tidak masuk
    grup (tidak punya varian AVG/MEDI lain).

    Mengembalikan tuple (df_baru, list_kolom_yang_dibuang) supaya keputusan
    ini terdokumentasi jelas di notebook, bukan hilang diam-diam.
    """
    if groups is None:
        groups = build_housing_groups()

    df = df.copy()
    cols_to_drop = []
    for base_name, variant_cols in groups.items():
        preferred_col = f"{base_name}{preferred_suffix}"
        for col in variant_cols:
            if col != preferred_col and col in df.columns:
                cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop)
    return df, cols_to_drop


# 4. Redundancy Check - Pearson & Spearman (Fitur Numerik)
"""Setelah housing dirampingkan, cek redundancy di SISA fitur numerik dengan
dua metode: Pearson (hubungan linear) dan Spearman (hubungan monoton/rank,
lebih toleran ke outlier). Pasangan dengan |r| > REDUNDANCY_THRESHOLD (0.9)
di SALAH SATU metode dilaporkan sebagai kandidat redundan untuk direview
manual - fungsi ini hanya MELAPORKAN, tidak otomatis membuang kolom."""


def compute_numeric_correlation_matrices(df, exclude_cols=("SK_ID_CURR", "TARGET")):
    """
    Hitung matriks korelasi Pearson dan Spearman untuk seluruh kolom numerik
    (kecuali ID/TARGET). Mengembalikan dict {"pearson": df, "spearman": df}.
    """
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]
    return {
        "pearson": df[numeric_cols].corr(method="pearson"),
        "spearman": df[numeric_cols].corr(method="spearman"),
    }


def find_redundant_numeric_pairs(corr_matrices, threshold=REDUNDANCY_THRESHOLD):
    """
    Filter pasangan kolom numerik dengan |korelasi| > threshold, dari
    Pearson ATAU Spearman. Mengembalikan DataFrame dengan kolom:
    feature_1, feature_2, pearson_corr, spearman_corr - diurutkan dari
    korelasi absolut tertinggi.

    Pasangan ini hanya dilaporkan, bukan otomatis dibuang - keputusan
    membuang salah satu tetap perlu direview manual di notebook (misalnya
    mempertahankan yang secara bisnis lebih mudah dijelaskan).
    """
    pearson_df = corr_matrices["pearson"]
    spearman_df = corr_matrices["spearman"]
    cols = pearson_df.columns

    records = []
    for col_a, col_b in combinations(cols, 2):
        p_corr = pearson_df.loc[col_a, col_b]
        s_corr = spearman_df.loc[col_a, col_b]
        if abs(p_corr) > threshold or abs(s_corr) > threshold:
            records.append({
                "feature_1": col_a,
                "feature_2": col_b,
                "pearson_corr": round(p_corr, 4),
                "spearman_corr": round(s_corr, 4),
            })

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        result_df["max_abs_corr"] = result_df[["pearson_corr", "spearman_corr"]].abs().max(axis=1)
        result_df = result_df.sort_values("max_abs_corr", ascending=False).drop(columns="max_abs_corr").reset_index(drop=True)

    return result_df


# 5. Redundancy Check - Cramer's V (Fitur Kategorikal)
"""Cramer's V mengukur kekuatan asosiasi antar dua variabel kategorikal
(rentang 0-1, analog korelasi untuk data non-numerik). Dipakai untuk cek
redundancy antar kolom kategorikal, misalnya OCCUPATION_TYPE vs
ORGANIZATION_TYPE yang secara konsep bisa jadi saling tumpang tindih."""


def cramers_v(col_a, col_b):
    """
    Hitung Cramer's V antara dua Series kategorikal menggunakan chi-square
    contingency table, dengan bias correction (Bergsma 2013) supaya tidak
    over-estimate asosiasi pada tabel dengan kategori/sampel kecil.
    """
    contingency_table = pd.crosstab(col_a, col_b)
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape

    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)

    denom = min((k_corr - 1), (r_corr - 1))
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2_corr / denom)


def compute_categorical_association_matrix(df, categorical_cols):
    """
    Hitung matriks Cramer's V untuk seluruh pasangan kolom kategorikal yang
    diberikan. Mengembalikan DataFrame matriks simetris (mirip correlation
    matrix, tapi untuk data kategorikal).
    """
    n = len(categorical_cols)
    result = pd.DataFrame(np.eye(n), index=categorical_cols, columns=categorical_cols)

    for col_a, col_b in combinations(categorical_cols, 2):
        v = cramers_v(df[col_a], df[col_b])
        result.loc[col_a, col_b] = v
        result.loc[col_b, col_a] = v

    return result.round(4)


# 6. VarianceThreshold - Evaluasi 18 Fitur Near-Zero Variance

def evaluate_variance_threshold(df, cols=NEAR_ZERO_VARIANCE_COLS, threshold=0.01):
    """
    Hitung variance setiap kolom near-zero-variance dan tandai mana yang di
    bawah threshold (kandidat dibuang). threshold=0.01 berarti kolom binary
    dengan proporsi kelas minoritas di bawah ~1% akan ditandai.

    Fit VarianceThreshold dilakukan dengan threshold=0.0 (hanya menolak
    kolom benar-benar konstan), BUKAN dengan `threshold` yang diminta.
    Ini karena VarianceThreshold.fit() akan melempar ValueError jika TIDAK
    ADA satu kolom pun yang lolos filter saat fit - kondisi yang ternyata
    benar terjadi pada dataset ini (seluruh 18 fitur near-zero variance
    berada di bawah 0.01). Perbandingan terhadap `threshold` sesungguhnya
    dilakukan secara manual di kolom below_threshold, agar variance tetap
    bisa dihitung dan dilaporkan meski seluruh kolom sangat rendah.

    Mengembalikan DataFrame dengan kolom: feature, variance, below_threshold
    - diurutkan dari variance terendah. Tidak ada kolom yang otomatis
    dibuang di sini; keputusan final tetap direview manual.
    """
    existing_cols = [c for c in cols if c in df.columns]
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(df[existing_cols])

    variances = selector.variances_
    result_df = pd.DataFrame({
        "feature": existing_cols,
        "variance": variances,
    })
    result_df["below_threshold"] = result_df["variance"] < threshold
    result_df = result_df.sort_values("variance").reset_index(drop=True)

    return result_df


# 7. Drop Fitur Numerik Redundan (Explicit, Hasil Review Manual)
"""Setelah find_redundant_numeric_pairs() melaporkan 12 pasangan kandidat redundan
(Section 4) dan diverifikasi manual di notebook (termasuk crosstab FLAG_EMP_PHONE
vs DAYS_EMPLOYED_SENTINEL), hanya 3 kolom yang disepakati untuk dibuang. Fungsi ini
TIDAK otomatis membuang berdasarkan threshold korelasi - daftar kolom bersifat
eksplisit/hardcoded agar keputusan buang selalu bisa ditelusuri alasannya, dan
supaya fitur dengan makna bisnis berbeda (misal AMT_CREDIT vs AMT_GOODS_PRICE)
tidak ikut terbuang hanya karena korelasinya tinggi."""


EXPLICIT_NUMERIC_DROP_REASONS = {
    "FLAG_EMP_PHONE": (
        "Near-perfect inverse relationship dengan DAYS_EMPLOYED_SENTINEL "
        "(99.9961% baris mengikuti pola inverse, 12 exceptions dari crosstab). "
        "DAYS_EMPLOYED_SENTINEL dipertahankan karena berasal dari sentinel "
        "employment (365243) yang sudah diverifikasi pada tahap cleaning, "
        "sehingga lebih interpretable."
    ),
    "OBS_60_CNT_SOCIAL_CIRCLE": (
        "Hampir duplikat OBS_30_CNT_SOCIAL_CIRCLE (Pearson 0.9985, "
        "Spearman 0.9973). OBS_30_CNT_SOCIAL_CIRCLE dipertahankan karena "
        "jendela waktu lebih pendek/lebih recent, dianggap lebih relevan "
        "untuk memprediksi risiko saat ini."
    ),
    "REGION_RATING_CLIENT": (
        "Redundan dengan REGION_RATING_CLIENT_W_CITY (Pearson 0.9508, "
        "Spearman 0.9500). Versi W_CITY dipertahankan karena lebih granular "
        "(mempertimbangkan rating tingkat kota, bukan hanya region)."
    ),
}


def drop_redundant_numeric_features(df):
    """
    Menghapus fitur numerik redundan berdasarkan keputusan eksplisit hasil
    review korelasi (Pearson & Spearman) pada tahap Feature Engineering.

    Kolom yang dihapus dan alasannya:
    - FLAG_EMP_PHONE: near-perfect inverse relationship dengan
      DAYS_EMPLOYED_SENTINEL (99.9961%, 12 exceptions). Bukan exact
      duplicate, tapi informasinya nyaris redundan.
    - OBS_60_CNT_SOCIAL_CIRCLE: hampir duplikat OBS_30_CNT_SOCIAL_CIRCLE;
      versi 30-day window dipertahankan karena lebih recent.
    - REGION_RATING_CLIENT: redundan dengan REGION_RATING_CLIENT_W_CITY
      yang lebih granular (mempertimbangkan rating kota).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe hasil tahap seleksi housing (biasanya berjumlah 91 kolom).

    Returns
    -------
    df_dropped : pd.DataFrame
        Dataframe baru (copy) tanpa kolom yang di-drop.
    dropped_cols : list[str]
        Daftar kolom yang benar-benar ditemukan dan dihapus dari df.
    """
    candidate_cols = list(EXPLICIT_NUMERIC_DROP_REASONS.keys())
    dropped_cols = [c for c in candidate_cols if c in df.columns]

    missing_cols = set(candidate_cols) - set(dropped_cols)
    if missing_cols:
        print(f"Peringatan: kolom berikut tidak ditemukan di df dan dilewati: {missing_cols}")

    df_dropped = df.drop(columns=dropped_cols).copy()
    return df_dropped, dropped_cols

    # 8. Korelasi Fitur Near-Zero Variance terhadap TARGET
"""VarianceThreshold hanya mengukur seberapa tersebar nilai suatu fitur, TIDAK
mengukur seberapa informatif fitur itu terhadap TARGET. Sebuah flag yang jarang
bernilai 1 (variance rendah) tetap bisa sangat diskriminatif terhadap default
risk. Karena itu, sebelum memutuskan buang 18 fitur near-zero variance, kita
perlu bukti tambahan berupa korelasi terhadap TARGET - fungsi ini hanya
MELAPORKAN, keputusan buang/pertahankan tetap direview manual di notebook."""


def compute_near_zero_variance_target_correlation(df, cols=NEAR_ZERO_VARIANCE_COLS, target_col="TARGET"):
    """
    Hitung korelasi Pearson setiap fitur near-zero-variance terhadap TARGET.

    Dipakai sebagai bukti pendukung kedua (selain variance dari
    evaluate_variance_threshold()) sebelum memutuskan buang fitur near-zero
    variance. Fitur dengan variance rendah TAPI korelasi target yang masih
    terlihat sebaiknya dipertahankan, karena variance rendah tidak selalu
    berarti tidak informatif - terutama pada dataset dengan TARGET imbalanced
    seperti project ini (TARGET=1 hanya 8.07%).

    Mengembalikan DataFrame dengan kolom: feature, corr_with_target,
    diurutkan dari korelasi absolut tertinggi ke terendah.
    """
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dataframe.")

    existing_cols = [c for c in cols if c in df.columns]
    records = []
    for col in existing_cols:
        corr_value = df[col].corr(df[target_col], method="pearson")
        records.append({
            "feature": col,
            "corr_with_target": round(corr_value, 4),
        })

    result_df = pd.DataFrame(records)
    result_df["abs_corr"] = result_df["corr_with_target"].abs()
    result_df = result_df.sort_values("abs_corr", ascending=False).drop(columns="abs_corr").reset_index(drop=True)

    return result_df

# 9. Drop Fitur Near-Zero Variance (Explicit, Hasil Review Variance + Target Correlation)
"""Setelah evaluate_variance_threshold() menandai seluruh 18 fitur di bawah
threshold 0.01, dan compute_near_zero_variance_target_correlation() menunjukkan
korelasi ke TARGET dapat diabaikan (maksimum |r|=0.0116, jauh di bawah kategori
"Weak" pada skala IV yang dipakai di EDA), seluruh 18 fitur disepakati untuk
dibuang. Keputusan ini berbasis DUA bukti kuantitatif sekaligus - bukan hanya
variance rendah - agar tidak membuang fitur yang mungkin tetap diskriminatif
meski jarang bervariasi."""


NEAR_ZERO_VARIANCE_DROP_REASON = (
    "Variance di bawah threshold 0.01 (near-constant) DAN korelasi Pearson "
    "terhadap TARGET dapat diabaikan (maksimum |r|=0.0116 dari FLAG_DOCUMENT_13/16, "
    "jauh di bawah kategori 'Weak' pada skala IV EDA). Kedua bukti konsisten "
    "menunjukkan fitur ini tidak memberikan informasi prediktif yang berarti."
)


def drop_near_zero_variance_features(df, cols=NEAR_ZERO_VARIANCE_COLS):
    """
    Menghapus 18 fitur near-zero variance (FLAG_MOBIL, FLAG_CONT_MOBILE,
    dan 16 FLAG_DOCUMENT_*) berdasarkan keputusan eksplisit hasil review
    variance (evaluate_variance_threshold) DAN korelasi ke TARGET
    (compute_near_zero_variance_target_correlation).

    Semua 18 kolom di NEAR_ZERO_VARIANCE_COLS dibuang karena TIDAK ADA
    satu pun yang menunjukkan sinyal cukup kuat terhadap TARGET untuk
    dipertahankan meski variance-nya rendah - lihat NEAR_ZERO_VARIANCE_DROP_REASON
    untuk detail bukti kuantitatifnya.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe hasil tahap drop_redundant_numeric_features()
        (biasanya berjumlah 88 kolom).

    Returns
    -------
    df_dropped : pd.DataFrame
        Dataframe baru (copy) tanpa 18 kolom near-zero variance.
    dropped_cols : list[str]
        Daftar kolom yang benar-benar ditemukan dan dihapus dari df.
    """
    dropped_cols = [c for c in cols if c in df.columns]

    missing_cols = set(cols) - set(dropped_cols)
    if missing_cols:
        print(f"Peringatan: kolom berikut tidak ditemukan di df dan dilewati: {missing_cols}")

    df_dropped = df.drop(columns=dropped_cols).copy()
    return df_dropped, dropped_cols

# 10. Feature Creation - Selisih & Rasio AMT_CREDIT vs AMT_GOODS_PRICE
"""AMT_CREDIT dan AMT_GOODS_PRICE dipertahankan keduanya di Section 7 karena
beda makna bisnis (jumlah pinjaman vs harga barang). Verifikasi di notebook
menunjukkan tidak ada AMT_GOODS_PRICE bernilai 0/negatif/null pada data training
(min=40.500), sehingga pembagian langsung aman untuk data ini. Fungsi ini tetap
menyertakan penanganan defensif (replace 0 jadi NaN sebelum membagi) untuk
mengantisipasi data inference/test di masa depan yang mungkin punya nilai 0,
bukan karena data training saat ini bermasalah."""


def create_credit_goods_features(df):
    """
    Buat dua fitur turunan dari AMT_CREDIT dan AMT_GOODS_PRICE untuk menangkap
    "pinjaman tambahan di luar harga barang":

    - AMT_CREDIT_MINUS_GOODS_PRICE: selisih absolut, menangkap besaran nominal
      kredit yang melebihi harga barang (bisa jadi biaya tambahan/asuransi/dsb).
    - AMT_CREDIT_TO_GOODS_RATIO: rasio, menangkap proporsi kredit relatif
      terhadap harga barang (misal rasio 1.2 berarti kredit 20% lebih besar
      dari harga barang).

    Pembagian dilakukan terhadap AMT_GOODS_PRICE yang sudah diganti 0 -> NaN
    (division by zero safety), bukan karena data training saat ini punya
    nilai 0 (sudah diverifikasi tidak ada), tapi sebagai antisipasi data
    inference di masa depan. Kolom original TIDAK di-overwrite.
    """
    df = df.copy()

    df["AMT_CREDIT_MINUS_GOODS_PRICE"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]

    safe_goods_price = df["AMT_GOODS_PRICE"].replace(0, np.nan)
    df["AMT_CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / safe_goods_price

    return df


# 10. Feature Creation - Selisih & Rasio AMT_CREDIT vs AMT_GOODS_PRICE
"""AMT_CREDIT dan AMT_GOODS_PRICE dipertahankan keduanya di Section 7 karena
beda makna bisnis (jumlah pinjaman vs harga barang). Verifikasi di notebook
menunjukkan tidak ada AMT_GOODS_PRICE bernilai 0/negatif/null pada data training
(min=40.500), sehingga pembagian langsung aman untuk data ini. Fungsi ini tetap
menyertakan penanganan defensif (replace 0 jadi NaN sebelum membagi) untuk
mengantisipasi data inference/test di masa depan.

LOG_AMT_CREDIT_TO_GOODS_RATIO ditambahkan khusus untuk jalur linear/MLP,
karena distribusi rasio menunjukkan ekor kanan panjang (min=0.15, 75%=1.198,
max=6.0). Memakai np.log() biasa (bukan log1p) karena rasio selalu positif
dan tidak pernah mendekati nol setelah guard divide-by-zero diterapkan.

AMT_CREDIT_MINUS_GOODS_PRICE TIDAK dibuatkan versi LOG karena nilainya bisa
negatif (AMT_CREDIT < AMT_GOODS_PRICE, indikasi uang muka), sehingga log(x)
tidak terdefinisi untuk sebagian data."""


def create_credit_goods_features(df):
    """
    Buat fitur turunan dari AMT_CREDIT dan AMT_GOODS_PRICE untuk menangkap
    "pinjaman tambahan di luar harga barang":

    - AMT_CREDIT_MINUS_GOODS_PRICE: selisih absolut. Dipakai di jalur
      tree-based DAN linear/MLP (aman, bukan kombinasi linear dari versi LOG
      AMT_CREDIT/AMT_GOODS_PRICE yang dipakai jalur linear).
    - AMT_CREDIT_TO_GOODS_RATIO: rasio raw. Dipakai HANYA di jalur tree-based.
    - LOG_AMT_CREDIT_TO_GOODS_RATIO: versi log dari rasio. Dipakai HANYA di
      jalur linear/MLP, untuk mengatasi skewness ekor kanan pada rasio raw.

    Pembagian dilakukan terhadap AMT_GOODS_PRICE yang sudah diganti 0 -> NaN
    (division by zero safety) sebelum dipakai baik untuk rasio raw maupun
    versi log-nya. Kolom original TIDAK di-overwrite.
    """
    df = df.copy()

    df["AMT_CREDIT_MINUS_GOODS_PRICE"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]

    safe_goods_price = df["AMT_GOODS_PRICE"].replace(0, np.nan)
    df["AMT_CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / safe_goods_price
    df["LOG_AMT_CREDIT_TO_GOODS_RATIO"] = np.log(df["AMT_CREDIT_TO_GOODS_RATIO"])

    return df

# 11. Final Feature List Split - Tree-based vs Linear/MLP
"""Dua jalur model butuh daftar fitur berbeda untuk menghindari redundansi:
tree-based memakai versi RAW kolom finansial (lebih mudah diinterpretasi,
tree tidak butuh distribusi simetris), linear/MLP memakai versi LOG (mengatasi
skewness, penting untuk model yang sensitif skala/distribusi). Pasangan
raw-vs-LOG didefinisikan eksplisit di FINANCIAL_LOG_PAIRS, mencakup 4 kolom
finansial dari cleaning.py PLUS AMT_CREDIT_TO_GOODS_RATIO yang baru dibuat di
Section 10. AMT_CREDIT_MINUS_GOODS_PRICE tidak masuk pasangan manapun karena
tidak punya versi LOG (bisa negatif), sehingga otomatis masuk ke KEDUA jalur."""


FINANCIAL_LOG_PAIRS = {
    "AMT_INCOME_TOTAL": "LOG_AMT_INCOME_TOTAL",
    "AMT_CREDIT": "LOG_AMT_CREDIT",
    "AMT_ANNUITY": "LOG_AMT_ANNUITY",
    "AMT_GOODS_PRICE": "LOG_AMT_GOODS_PRICE",
    "AMT_CREDIT_TO_GOODS_RATIO": "LOG_AMT_CREDIT_TO_GOODS_RATIO",
}


def build_final_feature_lists(df, exclude_cols=("SK_ID_CURR", "TARGET")):
    """
    Bangun dua daftar fitur final: tree-based dan linear/MLP.

    Aturan:
    - Jalur tree-based: exclude semua kolom LOG_* yang berpasangan dengan
      raw-nya (dari FINANCIAL_LOG_PAIRS), pertahankan versi raw.
    - Jalur linear/MLP: exclude semua kolom raw yang berpasangan dengan
      LOG-nya (dari FINANCIAL_LOG_PAIRS), pertahankan versi LOG.
    - Kolom lain (termasuk AMT_CREDIT_MINUS_GOODS_PRICE, seluruh kolom
      kategorikal, dan seluruh kolom numerik non-finansial) masuk ke
      KEDUA jalur karena tidak punya pasangan raw-LOG.
    - Encoding (One-Hot untuk non-CatBoost) dan scaling (StandardScaler
      untuk Logistic Regression/MLP) BELUM diterapkan di sini - itu
      dilakukan di tahap Modeling via Pipeline, setelah train-test split,
      untuk mencegah data leakage.

    Returns
    -------
    feature_lists : dict
        {"tree": list[str], "linear_mlp": list[str]}
    """
    all_cols = [c for c in df.columns if c not in exclude_cols]

    raw_cols = list(FINANCIAL_LOG_PAIRS.keys())
    log_cols = list(FINANCIAL_LOG_PAIRS.values())

    tree_features = [c for c in all_cols if c not in log_cols]
    linear_mlp_features = [c for c in all_cols if c not in raw_cols]

    return {"tree": tree_features, "linear_mlp": linear_mlp_features}

# 12. Simpan Dataset & Metadata Feature Engineering Final
"""Menutup tahap Feature Engineering dengan menyimpan dua artifact terpisah:
dataset hasil akhir (CSV) dan metadata keputusan (JSON). Dataset disimpan
dengan nama baru (application_train_featured.csv), TIDAK menimpa
application_train_clean.csv, supaya checkpoint cleaning tetap terjaga sebagai
titik kembali. Metadata disimpan sebagai JSON karena isinya berupa daftar
fitur dengan panjang berbeda-beda per jalur, bukan tabel baris-kolom seperti
CSV."""


import json


def save_feature_engineered_dataset(df, output_path):
    """
    Simpan dataframe hasil akhir Feature Engineering ke CSV baru.
    TIDAK menimpa application_train_clean.csv - file itu tetap jadi
    checkpoint cleaning yang terpisah dari hasil feature engineering.
    """
    df.to_csv(output_path, index=False)
    print(f"Dataset feature-engineered disimpan ke: {output_path}")
    print(f"Shape: {df.shape}")


def save_feature_engineering_metadata(
    dropped_housing_cols,
    dropped_numeric_cols,
    dropped_nzv_cols,
    feature_lists,
    output_path,
):
    """
    Simpan metadata keputusan Feature Engineering sebagai JSON: daftar fitur
    final per jalur (tree/linear_mlp), dan decision log seluruh kolom yang
    dibuang sepanjang tahap ini beserta alasannya - agar proses ini
    reproducible dan bisa ditelusuri ulang tanpa membaca ulang notebook.
    """
    metadata = {
        "tree_features": feature_lists["tree"],
        "linear_mlp_features": feature_lists["linear_mlp"],
        "n_tree_features": len(feature_lists["tree"]),
        "n_linear_mlp_features": len(feature_lists["linear_mlp"]),
        "decision_log": {
            "dropped_housing_variants": {
                "columns": dropped_housing_cols,
                "reason": "Redundan dengan representasi _MEDI (korelasi intra-grup > 0.96); "
                          "_MEDI dipilih karena median lebih robust terhadap outlier.",
            },
            "dropped_redundant_numeric": {
                "columns": dropped_numeric_cols,
                "reason": EXPLICIT_NUMERIC_DROP_REASONS,
            },
            "dropped_near_zero_variance": {
                "columns": dropped_nzv_cols,
                "reason": NEAR_ZERO_VARIANCE_DROP_REASON,
            },
            "financial_log_pairs": FINANCIAL_LOG_PAIRS,
        },
    }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata Feature Engineering disimpan ke: {output_path}")