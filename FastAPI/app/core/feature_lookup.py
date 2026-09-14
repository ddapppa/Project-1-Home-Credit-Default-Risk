# Bagian 1 — Import dan konstanta housing. Enam nilai median dan tiga default kategorikal ini saya jadikan 
# konstanta bernama dengan komentar sumbernya persis, supaya siapa pun yang baca kode (termasuk kamu sendiri 
# 6 bulan lagi) tahu ini bukan angka ajaib.



# FastAPI/app/core/feature_lookup.py
import json
import logging
from datetime import date
from functools import lru_cache


import numpy as np
import pandas as pd


from app.schemas.applicant import ApplicantModeNew
from app.core.config import get_settings
from app.core.exceptions import SKIDNotFoundError


logger = logging.getLogger("home_credit_api")
settings = get_settings()


# Median dari application_train_clean.csv, diambil 2026-09-04, dipakai cleaning.py
# saat imputasi nasabah tanpa data housing (lihat cleaning.py L150-152).
HOUSING_NUMERIC_DEFAULTS = {
    "FLOORSMAX_MEDI": 0.1667,
    "LIVINGAREA_MEDI": 0.0749,
    "APARTMENTS_MEDI": 0.0864,
    "YEARS_BEGINEXPLUATATION_MEDI": 0.9816,
    "ENTRANCES_MEDI": 0.1379,
    "TOTALAREA_MODE": 0.0688,
}
# Sesuai cleaning.py L154-156: kategori housing missing diisi string "Missing"
HOUSING_CATEGORICAL_DEFAULTS = {
    "WALLSMATERIAL_MODE": "Missing",
    "EMERGENCYSTATE_MODE": "Missing",
    "HOUSETYPE_MODE": "Missing",
}


# Prefix penanda kolom hasil agregasi v2/v3 (bukan fitur aplikasi v1)
AGGREGATE_PREFIXES = ("BUREAU_", "BB_", "PREV_", "POS_", "CC_", "INSTAL_")


# Bagian 2 — Field administratif/sistem yang bukan hasil wawancara petugas. Ini fitur yang ada di 
# training tapi tidak wajar ditanyakan manual (sesuai diskusi kita jauh sebelumnya) — saya isi dengan 
# nilai mayoritas/wajar, bukan NaN, karena di data asli field ini hampir selalu terisi 
# (bukan pola "informative missing" seperti housing).


# Bagian 3 — Load metadata & pipeline lookup, di-cache supaya tidak dibaca ulang tiap request. 
# tree_features_v3 dibaca dinamis dari file JSON, bukan hardcode 739 nama — sesuai keputusan kita sebelumnya.
@lru_cache
def load_tree_feature_columns() -> list[str]:
    with open(settings.resolved_feature_metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata["tree_features_v3"]



_demo_lookup_df: pd.DataFrame | None = None


def get_existing_applicant_features(sk_id_curr: int) -> pd.DataFrame:
    global _demo_lookup_df
    if _demo_lookup_df is None:
        # CATATAN PERUBAHAN (deployment Vercel): sumber data diganti dari
        # demo_lookup.parquet ke demo_lookup.csv.gz supaya dependency
        # `pyarrow` (~100MB) tidak perlu ikut di-deploy. Isi datanya identik,
        # hanya format file dan cara baca yang berubah.
        _demo_lookup_df = pd.read_csv(settings.resolved_demo_lookup_path, compression="gzip")


    row = _demo_lookup_df[_demo_lookup_df["SK_ID_CURR"] == sk_id_curr]
    if row.empty:
        raise SKIDNotFoundError(sk_id_curr)


    feature_cols = load_tree_feature_columns()
    return row[feature_cols].reset_index(drop=True)



# Bagian 4 — Membangun raw_map dari input Mode Baru. Ini jantung file ini: mengonversi field 
# form jadi nilai-nilai fitur v1 sesuai formula persis dari features.py/cleaning.py.
def _build_raw_application_map(applicant: ApplicantModeNew) -> dict:
    today = date.today()
    days_birth = -(today - applicant.tanggal_lahir).days


    if applicant.status_pekerjaan == "Tidak Bekerja":
        days_employed = np.nan
        days_employed_sentinel = 1
    else:
        days_employed = -(today - applicant.tanggal_mulai_kerja).days
        days_employed_sentinel = 0


    amt_goods_price = applicant.amt_goods_price
    if amt_goods_price and amt_goods_price > 0:
        amt_credit_minus_goods = applicant.amt_credit - amt_goods_price
        amt_credit_to_goods_ratio = applicant.amt_credit / amt_goods_price
    else:
        amt_credit_minus_goods = np.nan
        amt_credit_to_goods_ratio = np.nan


    own_car_age = applicant.own_car_age if applicant.flag_own_car == "Y" else np.nan


    raw_map = {
        "NAME_CONTRACT_TYPE": applicant.name_contract_type,
        "CODE_GENDER": applicant.code_gender,
        "FLAG_OWN_CAR": applicant.flag_own_car,
        "FLAG_OWN_REALTY": applicant.flag_own_realty,
        "CNT_CHILDREN": applicant.cnt_children,
        "AMT_INCOME_TOTAL": applicant.amt_income_total,
        "AMT_CREDIT": applicant.amt_credit,
        "AMT_ANNUITY": applicant.amt_annuity,
        "AMT_GOODS_PRICE": amt_goods_price if amt_goods_price else np.nan,
        "NAME_TYPE_SUITE": applicant.name_type_suite,
        "NAME_INCOME_TYPE": applicant.name_income_type,
        "NAME_EDUCATION_TYPE": applicant.name_education_type,
        "NAME_FAMILY_STATUS": applicant.name_family_status,
        "NAME_HOUSING_TYPE": applicant.name_housing_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "DAYS_EMPLOYED_SENTINEL": days_employed_sentinel,
        "OWN_CAR_AGE": own_car_age,
        "OCCUPATION_TYPE": applicant.occupation_type,
        "CNT_FAM_MEMBERS": applicant.cnt_fam_members,
        "ORGANIZATION_TYPE": applicant.organization_type,
        # EXT_SOURCE: Opsi B+C -> tidak ada data eksternal, NaN + flag missing
        "EXT_SOURCE_1": np.nan, "EXT_SOURCE_1_MISSING": 1,
        "EXT_SOURCE_2": np.nan,
        "EXT_SOURCE_3": np.nan, "EXT_SOURCE_3_MISSING": 1,
        "AMT_CREDIT_MINUS_GOODS_PRICE": amt_credit_minus_goods,
        "AMT_CREDIT_TO_GOODS_RATIO": amt_credit_to_goods_ratio,
        # Housing: Mode Baru = tidak ada data housing (konsisten dgn cleaning.py)
        "HAS_HOUSING_INFO": 0,
        **HOUSING_NUMERIC_DEFAULTS,
        **HOUSING_CATEGORICAL_DEFAULTS,
        **_get_misc_system_defaults(),
    }
    return raw_map


# Bagian 5 — Merangkai semua ke satu baris DataFrame 739 kolom. Untuk kolom agregasi v2/v3, 
# aturan _COUNT→0 dan lainnya→NaN diterapkan otomatis. Untuk kolom v1 yang tidak ada di raw_map 
# (kemungkinan field yang terlewat dari analisis kita), sistem tidak crash — tapi mencatat warning 
# supaya kamu tahu ada gap yang perlu ditambal.
def build_new_applicant_features(applicant: ApplicantModeNew) -> pd.DataFrame:
    feature_cols = load_tree_feature_columns()
    raw_map = _build_raw_application_map(applicant)


    row = {}
    unmapped_cols = []


    for col in feature_cols:
        if col in raw_map:
            row[col] = raw_map[col]
        elif col.startswith(AGGREGATE_PREFIXES):
            row[col] = 0 if col.endswith("_COUNT") else np.nan
        else:
            row[col] = np.nan
            unmapped_cols.append(col)


    if unmapped_cols:
        logger.warning(
            f"Mode Baru: {len(unmapped_cols)} kolom v1 tidak termapping, diisi NaN: {unmapped_cols}"
        )


    return pd.DataFrame([row])[feature_cols]


def _get_misc_system_defaults() -> dict:
    now = pd.Timestamp.now()
    weekday_map = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
    return {
        "FLAG_MOBIL": 1, "FLAG_EMP_PHONE": 1, "FLAG_WORK_PHONE": 0,
        "FLAG_CONT_MOBILE": 1, "FLAG_PHONE": 0, "FLAG_EMAIL": 0,
        "HOUR_APPR_PROCESS_START": now.hour,
        "WEEKDAY_APPR_PROCESS_START": weekday_map[now.weekday()],
        "REG_REGION_NOT_LIVE_REGION": 0, "REG_REGION_NOT_WORK_REGION": 0,
        "LIVE_REGION_NOT_WORK_REGION": 0, "REG_CITY_NOT_LIVE_CITY": 0,
        "REG_CITY_NOT_WORK_CITY": 0, "LIVE_CITY_NOT_WORK_CITY": 0,
        # BARU: median/mode dari application_train_clean.csv (missing asli 0%), dicek 2026-09-05
        "DAYS_LAST_PHONE_CHANGE": -757.0,
        "DAYS_ID_PUBLISH": -3254.0,
        "DAYS_REGISTRATION": -4504.0,
        "REGION_POPULATION_RELATIVE": 0.01885,
        "REGION_RATING_CLIENT_W_CITY": 2.0,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
        "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
        "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
        "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
        "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
        "OBS_30_CNT_SOCIAL_CIRCLE": 0.0,
        "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
        "FLAG_DOCUMENT_3": 1,   # mode: 71% nasabah submit dokumen ini
        "FLAG_DOCUMENT_5": 0,   # mode: 98.5% tidak submit
        "FLAG_DOCUMENT_6": 0,   # mode: 91.2% tidak submit
        "FLAG_DOCUMENT_8": 0,   # mode: 91.9% tidak submit
    }