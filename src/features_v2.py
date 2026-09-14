"""
src/features_v2.py

Modul reusable untuk Feature Engineering v2 (Iterasi 2): agregasi 6 tabel
tambahan Home Credit jadi fitur level SK_ID_CURR, digabung ke dataset utama.
notebooks/07_feature_engineering_v2.ipynb HANYA memanggil fungsi di modul ini.

Skema Opsi A (baseline): agregasi SERAGAM (mean/max/min/sum/count) untuk semua
kolom numerik, One-Hot + proporsi untuk kolom kategorikal. Opsi B (agregasi
disesuaikan per kolom) baru dikerjakan kalau skor v2 ini belum cukup naik.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import config
from cleaning_auxiliary import fix_sentinel_values

SENTINEL_COLUMNS_PREVIOUS_APPLICATION = [
    "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE", "DAYS_TERMINATION",
]
COLUMNS_TO_DROP_PREVIOUS_APPLICATION = ["RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED"]
DEFAULT_NUMERIC_AGGS = ["mean", "max", "min", "sum", "count"]

# Dua Fungsi Generik: Agregasi Numerik dan Kategorikal
# Ini jantung dari pendekatan Opsi A — satu fungsi dipakai berulang untuk semua tabel, bukan 
# fungsi terpisah per tabel (sama seperti prinsip cleaning_auxiliary.py sebelumnya).

def aggregate_numeric_features(df: pd.DataFrame, group_col: str, prefix: str, agg_funcs: list = None) -> pd.DataFrame:
    """
    Meringkas semua kolom numerik jadi satu baris per group_col, memakai fungsi
    agregasi yang SAMA (mean/max/min/sum/count) untuk semua kolom -- Opsi A.
    """
    agg_funcs = agg_funcs or DEFAULT_NUMERIC_AGGS
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != group_col and not c.startswith("SK_ID")]

    agg_df = df.groupby(group_col)[numeric_cols].agg(agg_funcs)
    agg_df.columns = [f"{prefix}_{col}_{stat}".upper() for col, stat in agg_df.columns]
    return agg_df.reset_index()


def aggregate_categorical_features(df: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    """
    One-Hot kolom kategorikal, lalu hitung PROPORSI (mean) per group_col --
    contoh: proporsi riwayat kredit yang statusnya 'Active' dari total riwayat klien.
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not categorical_cols:
        return pd.DataFrame({group_col: df[group_col].unique()})

    one_hot = pd.get_dummies(df[[group_col] + categorical_cols], columns=categorical_cols)
    agg_df = one_hot.groupby(group_col).mean()
    agg_df.columns = [f"{prefix}_{col}_RATIO".upper() for col in agg_df.columns]
    return agg_df.reset_index()

# Fungsi Khusus per Tabel
# bureau_balance butuh perlakuan dua langkah (agregasi ke SK_ID_BUREAU dulu, baru ke SK_ID_CURR lewat bureau), 
# sedangkan 4 tabel lain bisa langsung diagregasi ke SK_ID_CURR karena kolom itu sudah ada langsung di tabelnya.
def build_bureau_balance_features(bureau_balance: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    """Dua langkah: agregasi per SK_ID_BUREAU dulu, lalu gabung & agregasi lagi per SK_ID_CURR."""
    numeric_agg = aggregate_numeric_features(bureau_balance, "SK_ID_BUREAU", "BB")
    categorical_agg = aggregate_categorical_features(bureau_balance, "SK_ID_BUREAU", "BB")
    bb_per_bureau = numeric_agg.merge(categorical_agg, on="SK_ID_BUREAU", how="outer")

    bb_with_curr = bb_per_bureau.merge(bureau[["SK_ID_CURR", "SK_ID_BUREAU"]], on="SK_ID_BUREAU", how="inner")
    bb_feature_cols = [c for c in bb_with_curr.columns if c not in ("SK_ID_BUREAU", "SK_ID_CURR")]

    final = bb_with_curr.groupby("SK_ID_CURR")[bb_feature_cols].agg(["mean", "max"])
    final.columns = [f"{col}_{stat}".upper() for col, stat in final.columns]
    return final.reset_index()


def build_bureau_features(bureau: pd.DataFrame, bureau_balance_features: pd.DataFrame = None) -> pd.DataFrame:
    """Agregasi bureau.csv per SK_ID_CURR, opsional gabung fitur turunan bureau_balance."""
    numeric_agg = aggregate_numeric_features(bureau, "SK_ID_CURR", "BUREAU")
    categorical_agg = aggregate_categorical_features(bureau, "SK_ID_CURR", "BUREAU")
    result = numeric_agg.merge(categorical_agg, on="SK_ID_CURR", how="outer")

    if bureau_balance_features is not None:
        result = result.merge(bureau_balance_features, on="SK_ID_CURR", how="left")
    return result


def build_previous_application_features(previous_application: pd.DataFrame) -> pd.DataFrame:
    """Perbaiki sentinel (reuse dari cleaning_auxiliary), drop kolom nyaris kosong, lalu agregasi."""
    df = fix_sentinel_values(previous_application, SENTINEL_COLUMNS_PREVIOUS_APPLICATION)
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP_PREVIOUS_APPLICATION if c in df.columns])

    numeric_agg = aggregate_numeric_features(df, "SK_ID_CURR", "PREV")
    categorical_agg = aggregate_categorical_features(df, "SK_ID_CURR", "PREV")
    return numeric_agg.merge(categorical_agg, on="SK_ID_CURR", how="outer")


def build_pos_cash_features(pos_cash: pd.DataFrame) -> pd.DataFrame:
    """Agregasi POS_CASH_balance.csv langsung per SK_ID_CURR."""
    numeric_agg = aggregate_numeric_features(pos_cash, "SK_ID_CURR", "POS")
    categorical_agg = aggregate_categorical_features(pos_cash, "SK_ID_CURR", "POS")
    return numeric_agg.merge(categorical_agg, on="SK_ID_CURR", how="outer")


def build_credit_card_features(credit_card: pd.DataFrame) -> pd.DataFrame:
    """Agregasi credit_card_balance.csv langsung per SK_ID_CURR."""
    numeric_agg = aggregate_numeric_features(credit_card, "SK_ID_CURR", "CC")
    categorical_agg = aggregate_categorical_features(credit_card, "SK_ID_CURR", "CC")
    return numeric_agg.merge(categorical_agg, on="SK_ID_CURR", how="outer")


def build_installments_features(installments: pd.DataFrame) -> pd.DataFrame:
    """Agregasi installments_payments.csv langsung per SK_ID_CURR."""
    return aggregate_numeric_features(installments, "SK_ID_CURR", "INSTAL")

# Orkestrator: Gabungkan Semua Tabel + Merge ke Dataset Utama
def build_auxiliary_features_v2(tables: dict) -> pd.DataFrame:
    """Bangun fitur dari 6 tabel, gabung jadi satu DataFrame lebar (satu baris per SK_ID_CURR)."""
    bb_features = build_bureau_balance_features(tables["bureau_balance"], tables["bureau"])
    bureau_features = build_bureau_features(tables["bureau"], bb_features)
    prev_features = build_previous_application_features(tables["previous_application"])
    pos_features = build_pos_cash_features(tables["POS_CASH_balance"])
    cc_features = build_credit_card_features(tables["credit_card_balance"])
    instal_features = build_installments_features(tables["installments_payments"])

    feature_frames = [bureau_features, prev_features, pos_features, cc_features, instal_features]
    all_curr_ids = pd.concat([f[["SK_ID_CURR"]] for f in feature_frames]).drop_duplicates()

    result = all_curr_ids
    for f in feature_frames:
        result = result.merge(f, on="SK_ID_CURR", how="left")
    return result


def merge_with_main_dataset(df_main: pd.DataFrame, features_v2: pd.DataFrame, id_column: str = config.ID_COLUMN) -> pd.DataFrame:
    """
    Left join fitur v2 ke dataset utama -- SK_ID_CURR yang bukan milik train
    (orphan test-set, sudah didokumentasikan di tahap cleaning) otomatis tersaring di sini.
    """
    merged = df_main.merge(features_v2, on=id_column, how="left")

    count_cols = [c for c in features_v2.columns if c.endswith("_COUNT")]
    merged[count_cols] = merged[count_cols].fillna(0)
    return merged

def save_feature_metadata_v2(metadata: dict, output_path: Path = None) -> Path:
    """Menyimpan metadata fitur v2 (termasuk tree_features_v2 & linear_mlp_features_v2) ke JSON."""
    if output_path is None:
        output_path = config.PROCESSED_DATA_DIR / "feature_engineering_metadata_v2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return output_path


def build_feature_metadata_v2(df_v2: pd.DataFrame, metadata_v1: dict) -> dict:
    """
    Menggabungkan daftar fitur v1 (tree_features, linear_mlp_features) dengan
    508 fitur baru hasil agregasi 6 tabel tambahan. Fitur baru dipakai di KEDUA
    jalur karena sudah numerik murni (info kategorikal sudah jadi proporsi _RATIO).
    """
    exclude_cols = {config.ID_COLUMN, config.TARGET_COLUMN}
    all_v1_cols = set(metadata_v1["tree_features"]) | set(metadata_v1["linear_mlp_features"])
    new_feature_columns = [c for c in df_v2.columns if c not in exclude_cols and c not in all_v1_cols]

    return {
        "stage": "feature_engineering_v2",
        "aggregation_strategy": "Opsi A -- agregasi seragam (mean/max/min/sum/count) + one-hot proporsi kategorikal",
        "n_new_features": len(new_feature_columns),
        "new_feature_columns": new_feature_columns,
        "tree_features_v2": metadata_v1["tree_features"] + new_feature_columns,
        "linear_mlp_features_v2": metadata_v1["linear_mlp_features"] + new_feature_columns,
    }
