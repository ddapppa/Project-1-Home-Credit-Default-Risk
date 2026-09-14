"""
src/features_v3.py

Modul reusable untuk Feature Engineering v3 (Opsi B): menambahkan fitur rasio
custom dan fitur jendela waktu terkini (recent window) di atas fondasi v2.
notebooks/09_feature_engineering_v3.ipynb HANYA memanggil fungsi di modul ini.
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
from features_v2 import (
    build_auxiliary_features_v2, aggregate_numeric_features, merge_with_main_dataset,
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Rasio yang aman terhadap pembagian nol -- hasilnya NaN, bukan error/infinity."""
    denominator_safe = denominator.replace(0, np.nan)
    return numerator / denominator_safe

# Bagian 2 — Fitur Rasio per Tabel (Dihitung di Level Baris Mentah)
def add_ratio_features_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    DEBT_CREDIT_RATIO: proporsi utang terhadap total kredit per baris riwayat --
    menangkap beban utang RELATIF, bukan cuma nilai mentah yang bias oleh skala.
    """
    df = bureau.copy()
    df["DEBT_CREDIT_RATIO"] = _safe_ratio(df["AMT_CREDIT_SUM_DEBT"], df["AMT_CREDIT_SUM"])
    return df


def add_ratio_features_previous_application(previous_application: pd.DataFrame) -> pd.DataFrame:
    """
    APPLICATION_CREDIT_RATIO: kredit yang disetujui vs diajukan (<1 berarti Home
    Credit menurunkan jumlahnya -- sinyal kehati-hatian mereka di masa lalu).
    ANNUITY_CREDIT_RATIO: perkiraan lama cicilan (rasio kecil = cicilan panjang).
    """
    df = previous_application.copy()
    df["APPLICATION_CREDIT_RATIO"] = _safe_ratio(df["AMT_APPLICATION"], df["AMT_CREDIT"])
    df["ANNUITY_CREDIT_RATIO"] = _safe_ratio(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    return df


def add_ratio_features_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """
    PAYMENT_RATIO & PAYMENT_DIFF: seberapa dekat pembayaran aktual dengan yang
    seharusnya. DAYS_PAST_DUE: keterlambatan bayar dalam hari (positif = telat) --
    salah satu fitur paling berpengaruh di banyak solusi publik kompetisi ini.
    """
    df = installments.copy()
    df["PAYMENT_RATIO"] = _safe_ratio(df["AMT_PAYMENT"], df["AMT_INSTALMENT"])
    df["PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]
    df["DAYS_PAST_DUE"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    return df


def add_ratio_features_credit_card(credit_card: pd.DataFrame) -> pd.DataFrame:
    """BALANCE_LIMIT_RATIO: utilisasi kartu kredit -- fitur klasik dalam credit scoring."""
    df = credit_card.copy()
    df["BALANCE_LIMIT_RATIO"] = _safe_ratio(df["AMT_BALANCE"], df["AMT_CREDIT_LIMIT_ACTUAL"])
    return df

# Bagian 3 — Fitur Recent Window (Kondisi 6 Bulan Terakhir)
def build_recent_window_features(
    df: pd.DataFrame, group_col: str, prefix: str, months_col: str = "MONTHS_BALANCE", window: int = 6,
) -> pd.DataFrame:
    """
    Mengagregasi HANYA baris dalam window bulan terakhir (default 6 bulan) --
    menangkap kondisi TERKINI klien, beda dari agregasi v2 yang meratakan seluruh
    histori dan bisa menutupi perbaikan/perburukan kondisi baru-baru ini.
    """
    recent_df = df[df[months_col] >= -window]
    return aggregate_numeric_features(recent_df, group_col, f"{prefix}_RECENT{window}M")

# Bagian 4 — Orkestrator Utama v3
def build_features_v3(tables: dict) -> pd.DataFrame:
    """
    Enrich tabel mentah dengan fitur rasio (otomatis ikut teragregasi lewat
    build_auxiliary_features_v2 yang sudah ada), lalu tambahkan fitur
    recent-window untuk POS_CASH_balance dan credit_card_balance.
    """
    tables_enriched = dict(tables)
    tables_enriched["bureau"] = add_ratio_features_bureau(tables["bureau"])
    tables_enriched["previous_application"] = add_ratio_features_previous_application(tables["previous_application"])
    tables_enriched["installments_payments"] = add_ratio_features_installments(tables["installments_payments"])
    tables_enriched["credit_card_balance"] = add_ratio_features_credit_card(tables["credit_card_balance"])

    features_v3_base = build_auxiliary_features_v2(tables_enriched)

    pos_recent = build_recent_window_features(tables["POS_CASH_balance"], "SK_ID_CURR", "POS")
    cc_recent = build_recent_window_features(tables_enriched["credit_card_balance"], "SK_ID_CURR", "CC")

    features_v3 = features_v3_base.merge(pos_recent, on="SK_ID_CURR", how="left")
    features_v3 = features_v3.merge(cc_recent, on="SK_ID_CURR", how="left")
    return features_v3

# Bagian 5 — Simpan Dataset dan Metadata v3
def save_featured_dataset_v3(df: pd.DataFrame, output_path: Path = None) -> Path:
    """Menyimpan dataset final v3 ke CSV."""
    if output_path is None:
        output_path = config.PROCESSED_DATA_DIR / "application_train_featured_v3.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def build_feature_metadata_v3(df_v3: pd.DataFrame, metadata_v2: dict) -> dict:
    """Menggabungkan daftar fitur v2 dengan fitur baru v3 (rasio + recent window)."""
    exclude_cols = {config.ID_COLUMN, config.TARGET_COLUMN}
    all_v2_cols = set(metadata_v2["tree_features_v2"]) | set(metadata_v2["linear_mlp_features_v2"])
    new_feature_columns = [c for c in df_v3.columns if c not in exclude_cols and c not in all_v2_cols]

    return {
        "stage": "feature_engineering_v3",
        "approach": (
            "Opsi B -- fitur rasio custom (debt/credit, application/credit, "
            "annuity/credit, payment ratio, days past due, balance/limit) "
            "+ recent-window 6 bulan (POS_CASH_balance, credit_card_balance)"
        ),
        "n_new_features": len(new_feature_columns),
        "new_feature_columns": new_feature_columns,
        "tree_features_v3": metadata_v2["tree_features_v2"] + new_feature_columns,
        "linear_mlp_features_v3": metadata_v2["linear_mlp_features_v2"] + new_feature_columns,
    }


def save_feature_metadata_v3(metadata: dict, output_path: Path = None) -> Path:
    """Menyimpan metadata fitur v3 ke JSON."""
    if output_path is None:
        output_path = config.PROCESSED_DATA_DIR / "feature_engineering_metadata_v3.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return output_path

