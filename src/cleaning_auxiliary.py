"""
src/cleaning_auxiliary.py

Modul reusable untuk audit kualitas data 6 tabel tambahan Home Credit
(bureau, bureau_balance, previous_application, POS_CASH_balance,
credit_card_balance, installments_payments) sebelum masuk ke Feature
Engineering v2. notebooks/06_cleaning_auxiliary.ipynb HANYA memanggil
fungsi di modul ini.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import config

SENTINEL_VALUE = 365243  # nilai boneka yang dikonfirmasi dipakai lintas tabel di dataset Home Credit

TABLE_CONFIGS = {
    "bureau": {
        "filename": "bureau.csv",
        "id_columns": ["SK_ID_CURR", "SK_ID_BUREAU"],
        "referential_checks": [("SK_ID_CURR", "application_train")],
        "amount_columns": [
            "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY",
        ],
        "days_columns": [
            "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE",
            "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE",
        ],
    },
    "bureau_balance": {
        "filename": "bureau_balance.csv",
        "id_columns": ["SK_ID_BUREAU"],
        "referential_checks": [("SK_ID_BUREAU", "bureau")],
        "amount_columns": [],
        "days_columns": ["MONTHS_BALANCE"],
    },
    "previous_application": {
        "filename": "previous_application.csv",
        "id_columns": ["SK_ID_CURR", "SK_ID_PREV"],
        "referential_checks": [("SK_ID_CURR", "application_train")],
        "amount_columns": ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE"],
        "days_columns": [
            "DAYS_DECISION", "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
            "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION",
        ],
    },
    "POS_CASH_balance": {
        "filename": "POS_CASH_balance.csv",
        "id_columns": ["SK_ID_CURR", "SK_ID_PREV"],
        "referential_checks": [("SK_ID_CURR", "application_train"), ("SK_ID_PREV", "previous_application")],
        "amount_columns": [],
        "days_columns": ["MONTHS_BALANCE"],
    },
    "credit_card_balance": {
        "filename": "credit_card_balance.csv",
        "id_columns": ["SK_ID_CURR", "SK_ID_PREV"],
        "referential_checks": [("SK_ID_CURR", "application_train"), ("SK_ID_PREV", "previous_application")],
        "amount_columns": [
            "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT",
            "AMT_DRAWINGS_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT", "AMT_DRAWINGS_POS_CURRENT",
            "AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT",
        ],
        "days_columns": ["MONTHS_BALANCE"],
    },
    "installments_payments": {
        "filename": "installments_payments.csv",
        "id_columns": ["SK_ID_CURR", "SK_ID_PREV"],
        "referential_checks": [("SK_ID_CURR", "application_train"), ("SK_ID_PREV", "previous_application")],
        "amount_columns": ["AMT_INSTALMENT", "AMT_PAYMENT"],
        "days_columns": ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"],
    },
}

# Fungsi Load Data
def load_auxiliary_table(table_name: str, raw_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Memuat satu tabel tambahan mentah berdasarkan nama di TABLE_CONFIGS."""
    filename = TABLE_CONFIGS[table_name]["filename"]
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"File {filename} tidak ditemukan di {path}.")
    return pd.read_csv(path)


def load_all_auxiliary_tables(raw_dir: Path = config.RAW_DATA_DIR) -> dict:
    """Memuat keenam tabel tambahan sekaligus, dikembalikan sebagai dict {nama: DataFrame}."""
    return {name: load_auxiliary_table(name, raw_dir) for name in TABLE_CONFIGS}

#  Fungsi Profiling Dasar (Generik untuk Semua Tabel)
def profile_table(df: pd.DataFrame, table_name: str) -> dict:
    """Ringkasan dasar: shape, jumlah duplikat, dan persentase missing per kolom."""
    missing_pct = (df.isna().mean() * 100).round(2)
    return {
        "table_name": table_name,
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_duplicate_rows": int(df.duplicated().sum()),
        "columns_with_missing": missing_pct[missing_pct > 0].sort_values(ascending=False).to_dict(),
    }

#  Cek Integritas Referensial (Antar Tabel)
def check_referential_integrity(
    child_df: pd.DataFrame, child_id_col: str, parent_ids: set, parent_table_name: str
) -> dict:
    """
    Mengecek apakah semua nilai child_id_col di child_df ada di parent_ids (ID valid
    milik tabel induk). ID yang tidak ditemukan disebut 'orphan' -- tanda data tidak sinkron.
    """
    child_ids = set(child_df[child_id_col].dropna().unique())
    orphan_ids = child_ids - parent_ids
    return {
        "child_id_column": child_id_col,
        "parent_table": parent_table_name,
        "n_unique_child_ids": len(child_ids),
        "n_orphan_ids": len(orphan_ids),
        "pct_orphan": round(len(orphan_ids) / len(child_ids) * 100, 4) if child_ids else 0.0,
    }

# Deteksi Sentinel Value dan Nilai Negatif Tidak Logis
def detect_sentinel_values(df: pd.DataFrame, days_columns: list, sentinel: int = SENTINEL_VALUE) -> dict:
    """Menghitung berapa banyak baris di tiap kolom DAYS_*/MONTHS_* yang bernilai sentinel (365243)."""
    result = {}
    for col in days_columns:
        if col in df.columns:
            count = int((df[col] == sentinel).sum())
            if count > 0:
                result[col] = {"count": count, "pct": round(count / len(df) * 100, 4)}
    return result


def detect_illogical_negative_amounts(df: pd.DataFrame, amount_columns: list) -> dict:
    """Mengecek kolom AMT_* yang seharusnya non-negatif tapi ternyata ada nilai minus."""
    result = {}
    for col in amount_columns:
        if col in df.columns:
            count = int((df[col] < 0).sum())
            if count > 0:
                result[col] = {"count": count, "pct": round(count / len(df) * 100, 4)}
    return result

# Cek Tipe Data (Antisipasi Risiko dari Excel)
def check_dtype_consistency(df: pd.DataFrame, expected_numeric_cols: list) -> dict:
    """Memastikan kolom yang seharusnya numerik benar-benar bertipe numerik, bukan object/string."""
    problems = {}
    for col in expected_numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            problems[col] = str(df[col].dtype)
    return problems

# Orkestrator: Menjalankan Semua Cek untuk Satu Tabel
def run_quality_check(table_name: str, df: pd.DataFrame, parent_id_sets: dict) -> dict:
    """
    Menjalankan seluruh cek (profiling, duplikat, integritas referensial, sentinel,
    negatif tidak logis, dtype) untuk SATU tabel, memakai konfigurasi dari TABLE_CONFIGS.
    """
    cfg = TABLE_CONFIGS[table_name]

    report = {"profile": profile_table(df, table_name)}

    referential_results = []
    for child_col, parent_table in cfg["referential_checks"]:
        parent_ids = parent_id_sets[parent_table]
        referential_results.append(check_referential_integrity(df, child_col, parent_ids, parent_table))
    report["referential_integrity"] = referential_results

    report["sentinel_values"] = detect_sentinel_values(df, cfg["days_columns"])
    report["illogical_negative_amounts"] = detect_illogical_negative_amounts(df, cfg["amount_columns"])
    report["dtype_problems"] = check_dtype_consistency(df, cfg["amount_columns"] + cfg["days_columns"])

    return report

# Orkestrator: Menjalankan Semua Cek untuk Satu Tabel
def run_quality_check(table_name: str, df: pd.DataFrame, parent_id_sets: dict) -> dict:
    """
    Menjalankan seluruh cek (profiling, duplikat, integritas referensial, sentinel,
    negatif tidak logis, dtype) untuk SATU tabel, memakai konfigurasi dari TABLE_CONFIGS.
    """
    cfg = TABLE_CONFIGS[table_name]

    report = {"profile": profile_table(df, table_name)}

    referential_results = []
    for child_col, parent_table in cfg["referential_checks"]:
        parent_ids = parent_id_sets[parent_table]
        referential_results.append(check_referential_integrity(df, child_col, parent_ids, parent_table))
    report["referential_integrity"] = referential_results

    report["sentinel_values"] = detect_sentinel_values(df, cfg["days_columns"])
    report["illogical_negative_amounts"] = detect_illogical_negative_amounts(df, cfg["amount_columns"])
    report["dtype_problems"] = check_dtype_consistency(df, cfg["amount_columns"] + cfg["days_columns"])

    return report

# Menjalankan untuk Keenam Tabel Sekaligus (Loop)
def run_quality_check_all_tables(tables: dict, application_train_ids: set) -> dict:
    """
    Loop menjalankan run_quality_check untuk keenam tabel. parent_id_sets dibangun
    bertahap: SK_ID_CURR dari application_train, lalu SK_ID_BUREAU dari bureau,
    lalu SK_ID_PREV dari previous_application -- supaya cek integritas berjenjang benar.
    """
    parent_id_sets = {"application_train": application_train_ids}

    if "bureau" in tables:
        parent_id_sets["bureau"] = set(tables["bureau"]["SK_ID_BUREAU"].dropna().unique())
    if "previous_application" in tables:
        parent_id_sets["previous_application"] = set(tables["previous_application"]["SK_ID_PREV"].dropna().unique())

    all_reports = {}
    for table_name, df in tables.items():
        all_reports[table_name] = run_quality_check(table_name, df, parent_id_sets)
    return all_reports

# Simpan Laporan ke CSV
def save_quality_report(all_reports: dict, output_path: Path = None) -> Path:
    """Meratakan (flatten) hasil audit jadi satu tabel ringkas, simpan ke reports/."""
    if output_path is None:
        output_path = config.REPORTS_DIR / "auxiliary_data_quality_report.csv"

    rows = []
    for table_name, report in all_reports.items():
        profile = report["profile"]
        rows.append({
            "table_name": table_name,
            "n_rows": profile["n_rows"],
            "n_columns": profile["n_columns"],
            "n_duplicate_rows": profile["n_duplicate_rows"],
            "n_columns_with_missing": len(profile["columns_with_missing"]),
            "worst_missing_column": max(profile["columns_with_missing"], key=profile["columns_with_missing"].get)
                if profile["columns_with_missing"] else None,
            "worst_missing_pct": max(profile["columns_with_missing"].values())
                if profile["columns_with_missing"] else 0.0,
            "max_orphan_pct": max((r["pct_orphan"] for r in report["referential_integrity"]), default=0.0),
            "n_sentinel_columns_found": len(report["sentinel_values"]),
            "n_illogical_negative_columns": len(report["illogical_negative_amounts"]),
            "n_dtype_problems": len(report["dtype_problems"]),
        })

    report_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    return output_path

# Fungsi ini mengganti nilai boneka 365243 jadi NaN di kolom yang ditentukan.
# Sengaja mengembalikan copy baru (bukan mengubah df yang asli langsung) — 
# supaya kalau ada kesalahan di tahap selanjutnya, data mentah aslinya di memory tidak ikut rusak.
def fix_sentinel_values(df: pd.DataFrame, columns: list, sentinel: int = SENTINEL_VALUE) -> pd.DataFrame:
    """
    Mengganti nilai sentinel (365243) menjadi NaN pada kolom yang ditentukan.
    Mengembalikan DataFrame baru (copy) -- tidak mengubah df asli secara langsung.
    """
    df_fixed = df.copy()
    for col in columns:
        if col in df_fixed.columns:
            df_fixed[col] = df_fixed[col].replace(sentinel, np.nan)
    return df_fixed

# Fungsi ini menyimpan seluruh keputusan cleaning (apa yang ditemukan, apa yang diputuskan, dan kenapa) ke 
# satu file JSON
# mengikuti pola yang sama seperti feature_engineering_metadata.json yang sudah ada dari tahap sebelumnya.
def save_cleaning_decision_log(decisions: dict, output_path: Path = None) -> Path:
    """Menyimpan catatan keputusan cleaning tabel tambahan ke JSON, mirip feature_engineering_metadata.json."""
    if output_path is None:
        output_path = config.PROCESSED_DATA_DIR / "auxiliary_cleaning_decision_log.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, default=str)
    return output_path