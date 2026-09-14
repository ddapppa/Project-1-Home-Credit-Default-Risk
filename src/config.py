# Sumber Tunggal Path dan Konstanta
# Tujuan kode: File ini jadi satu-satunya tempat mendefinisikan path folder, nama file artifact, 
# dan konstanta global (RANDOM_SEED, TEST_SIZE, N_SPLITS, dll). Saya pakai pathlib.Path(__file__) 
# supaya path dihitung otomatis relatif terhadap lokasi file ini, sehingga tetap benar baik 
# dijalankan dari notebook di folder notebooks/ maupun dipanggil langsung sebagai modul — tidak 
# perlu hardcode D:\Kerjaan\Home Creadit\... yang rawan salah kalau projectnya dipindah folder 
# atau dibuka di laptop lain.

"""
src/config.py
Sumber tunggal (single source of truth) untuk path folder dan konstanta
global project Home Credit Default Risk. Semua modul lain (cleaning,
features, modeling) WAJIB mengambil path/konstanta dari sini, bukan
mendefinisikan ulang.
"""

from pathlib import Path

# --- Root project dihitung otomatis dari lokasi file ini ---
# config.py ada di: <project_root>/src/config.py
# .parent          -> <project_root>/src
# .parent.parent   -> <project_root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Struktur folder utama ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Path dataset & artifact yang sudah ada (dari tahap sebelumnya) ---
FEATURED_DATASET_PATH = PROCESSED_DATA_DIR / "application_train_featured.csv"
FEATURE_METADATA_PATH = PROCESSED_DATA_DIR / "feature_engineering_metadata.json"

# --- Path artifact BARU yang akan dibuat di tahap Modeling ---
HOLDOUT_SPLIT_PATH = PROCESSED_DATA_DIR / "holdout_split.json"
CV_FOLDS_PATH = PROCESSED_DATA_DIR / "cv_folds.json"
MODEL_SUMMARY_PATH = REPORTS_DIR / "model_summary.csv"

# --- Kolom identitas & target ---
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"

# --- Konstanta reproducibility & validasi ---
RANDOM_SEED = 42
TEST_SIZE = 0.2          # 20% holdout test, tidak boleh disentuh sebelum model final dipilih
N_SPLITS = 5              # Stratified 5-Fold CV pada 80% development set
PRIMARY_METRIC = "roc_auc"

# --- Konstanta preprocessing kategorikal ---
RARE_CATEGORY_THRESHOLD = 50   # kategori dengan frekuensi < 50 di fold-train digabung jadi "Other"