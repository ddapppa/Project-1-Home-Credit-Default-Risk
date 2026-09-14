"""
Layer explainability berbasis SHAP untuk memberi alasan spesifik di balik
setiap prediksi Mode Existing (top faktor pendorong risiko naik/turun).

PENTING:
1. Hanya dipanggil untuk Mode Existing.
2. CODE_GENDER dikecualikan dari tampilan (protected characteristic).
3. Label TIDAK menyertakan kata sifat "tinggi"/"rendah" — nilai aktual
   fitur ditampilkan langsung (value_display), arah risiko sudah
   terwakili lewat pengelompokan (hijau/merah) di frontend.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import shap

logger = logging.getLogger("home_credit_api")

TOP_N_REASONS = 5
MIN_CONTRIBUTION_PCT = 1.5

EXCLUDED_FROM_DISPLAY = {
    "CODE_GENDER",
}

RAW_FEATURE_LABELS_ID = {
    "EXT_SOURCE_1": "Skor riwayat kredit dari lembaga eksternal 1",
    "EXT_SOURCE_2": "Skor riwayat kredit dari lembaga eksternal 2",
    "EXT_SOURCE_3": "Skor riwayat kredit dari lembaga eksternal 3",
    "DAYS_BIRTH": "Usia nasabah",
    "DAYS_EMPLOYED": "Lama masa kerja nasabah",
    "DAYS_REGISTRATION": "Lama waktu sejak nasabah terdaftar",
    "DAYS_ID_PUBLISH": "Lama waktu sejak dokumen identitas diterbitkan",
    "DAYS_LAST_PHONE_CHANGE": "Lama waktu sejak nomor telepon terakhir diubah",
    "AMT_INCOME_TOTAL": "Total penghasilan nasabah",
    "AMT_CREDIT": "Jumlah kredit yang diajukan",
    "AMT_ANNUITY": "Besar angsuran per periode",
    "AMT_GOODS_PRICE": "Harga barang yang dibeli",
    "CNT_CHILDREN": "Jumlah anak",
    "CNT_FAM_MEMBERS": "Jumlah anggota keluarga",
    "NAME_EDUCATION_TYPE": "Tingkat pendidikan nasabah",
    "NAME_FAMILY_STATUS": "Status pernikahan nasabah",
    "NAME_HOUSING_TYPE": "Jenis tempat tinggal nasabah",
    "NAME_INCOME_TYPE": "Jenis sumber penghasilan nasabah",
    "NAME_CONTRACT_TYPE": "Jenis kontrak kredit yang diajukan",
    "OCCUPATION_TYPE": "Jenis pekerjaan nasabah",
    "ORGANIZATION_TYPE": "Jenis organisasi/perusahaan tempat bekerja",
    "FLAG_OWN_CAR": "Status kepemilikan mobil",
    "FLAG_OWN_REALTY": "Status kepemilikan rumah/tanah",
    "OWN_CAR_AGE": "Usia mobil yang dimiliki",
    "REGION_POPULATION_RELATIVE": "Kepadatan populasi wilayah tempat tinggal",
    "REGION_RATING_CLIENT": "Peringkat wilayah tempat tinggal nasabah",
    "REGION_RATING_CLIENT_W_CITY": "Peringkat wilayah kota tempat tinggal nasabah",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Jumlah kerabat/relasi yang pernah menunggak 30 hari",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Jumlah kerabat/relasi yang pernah menunggak 60 hari",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Jumlah kerabat/relasi yang diamati (30 hari)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Jumlah permintaan cek kredit dalam 1 tahun terakhir",
    "AMT_REQ_CREDIT_BUREAU_MON": "Jumlah permintaan cek kredit dalam 1 bulan terakhir",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Jumlah permintaan cek kredit dalam 1 kuartal terakhir",
    "FLAG_DOCUMENT_3": "Kelengkapan dokumen pendukung nasabah",
    "NAME_TYPE_SUITE": "Pendamping nasabah saat mengajukan kredit",
}

ENGINEERED_FEATURE_LABELS = {
    "AMT_CREDIT_MINUS_GOODS_PRICE": "Selisih jumlah kredit dengan harga barang",
    "AMT_CREDIT_TO_GOODS_RATIO": "Rasio jumlah kredit terhadap harga barang",
    "DAYS_EMPLOYED_SENTINEL": "Data lama bekerja yang tidak wajar/anomali",
    "HAS_HOUSING_INFO": "Ketersediaan data informasi hunian",
    "EXT_SOURCE_1_MISSING": "Ketersediaan skor riwayat kredit eksternal 1",
    "EXT_SOURCE_2_MISSING": "Ketersediaan skor riwayat kredit eksternal 2",
    "EXT_SOURCE_3_MISSING": "Ketersediaan skor riwayat kredit eksternal 3",
    "BUREAU_DEBT_CREDIT_RATIO_MEAN": "Rata-rata rasio utang terhadap kredit dari riwayat biro kredit",
    "BUREAU_DEBT_CREDIT_RATIO_MAX": "Rasio utang terhadap kredit tertinggi dari riwayat biro kredit",
    "INSTAL_DAYS_PAST_DUE_MEAN": "Rata-rata keterlambatan pembayaran cicilan sebelumnya",
    "INSTAL_DAYS_PAST_DUE_MAX": "Keterlambatan pembayaran cicilan yang paling lama",
    "INSTAL_AMT_PAYMENT_SUM": "Total pembayaran cicilan pada riwayat sebelumnya",
    "INSTAL_PAYMENT_DIFF_MEAN": "Rata-rata selisih pembayaran seharusnya vs aktual pada cicilan sebelumnya",
    "POS_RECENT6M_SK_DPD_MEAN": "Rata-rata keterlambatan cicilan barang 6 bulan terakhir",
    "POS_CNT_INSTALMENT_FUTURE_MEAN": "Rata-rata sisa jumlah cicilan barang yang belum lunas",
    "CC_RECENT6M_AMT_BALANCE_MEAN": "Rata-rata saldo kartu kredit 6 bulan terakhir",
    "CC_RECENT6M_BALANCE_LIMIT_RATIO_MAX": "Rasio saldo terhadap limit kartu kredit tertinggi (6 bulan terakhir)",
    "PREV_AMT_CREDIT_MEAN": "Rata-rata jumlah kredit pada pengajuan sebelumnya",
    "PREV_DAYS_LAST_DUE_MAX": "Keterlambatan hari jatuh tempo terakhir pada kredit sebelumnya",
    "PREV_NAME_CONTRACT_STATUS_REFUSED_RATIO": "Proporsi pengajuan kredit sebelumnya yang ditolak",
    "PREV_NAME_GOODS_CATEGORY_TOURISM_RATIO": "Proporsi pengajuan sebelumnya untuk kategori barang pariwisata",
}

CURRENCY_FEATURES = {
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "AMT_CREDIT_MINUS_GOODS_PRICE",
    "CC_RECENT6M_AMT_BALANCE_MEAN",
    "PREV_AMT_CREDIT_MEAN",
    "INSTAL_AMT_PAYMENT_SUM",
    "INSTAL_PAYMENT_DIFF_MEAN",
}

COUNT_FEATURE_PREFIXES = ("AMT_REQ_CREDIT_BUREAU_",)

LARGE_SCALE_DAYS_FEATURES = {
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
}


def _humanize_fallback(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def get_feature_label(feature_name: str) -> str:
    base_name = feature_name
    for suffix in ("_MEAN", "_MAX", "_MIN", "_SUM", "_COUNT", "_RATIO", "_MISSING", "_RECENT6M"):
        base_name = base_name.replace(suffix, "")

    if feature_name in ENGINEERED_FEATURE_LABELS:
        return ENGINEERED_FEATURE_LABELS[feature_name]
    if feature_name in RAW_FEATURE_LABELS_ID:
        return RAW_FEATURE_LABELS_ID[feature_name]
    if base_name in RAW_FEATURE_LABELS_ID:
        return RAW_FEATURE_LABELS_ID[base_name]

    return _humanize_fallback(feature_name)


def _format_value(feature_name: str, value) -> str:
    """Format nilai asli fitur jadi teks yang mudah dibaca pegawai."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "tidak tersedia"

    if feature_name == "DAYS_BIRTH":
        usia = abs(float(value)) / 365.25
        return f"{usia:.0f} tahun"

    if feature_name == "DAYS_EMPLOYED":
        if float(value) == 365243:
            return "tidak bekerja / data tidak wajar"
        lama = abs(float(value)) / 365.25
        return f"{lama:.1f} tahun"

    if feature_name in LARGE_SCALE_DAYS_FEATURES:
        lama = abs(float(value)) / 365.25
        return f"{lama:.1f} tahun yang lalu"

    if "DAYS" in feature_name or "DPD" in feature_name:
        try:
            return f"{float(value):.0f} hari"
        except (TypeError, ValueError):
            return str(value)

    if feature_name.startswith("EXT_SOURCE") and "_MISSING" not in feature_name:
        try:
            return f"{float(value):.2f} (skala 0-1)"
        except (TypeError, ValueError):
            return str(value)

    if feature_name in CURRENCY_FEATURES:
        try:
            return f"Rp {float(value):,.0f}".replace(",", ".")
        except (TypeError, ValueError):
            return str(value)

    if feature_name.startswith(COUNT_FEATURE_PREFIXES):
        try:
            return f"{int(value)} kali"
        except (TypeError, ValueError):
            return str(value)

    if "RATIO" in feature_name:
        try:
            return f"{float(value):.2f}x"
        except (TypeError, ValueError):
            return str(value)

    if isinstance(value, bool):
        return "Ya" if value else "Tidak"

    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.2f}"

    return str(value)


_explainer_cache: dict[int, Any] = {}


def _get_explainer(model):
    key = id(model)
    if key not in _explainer_cache:
        logger.info("Membangun SHAP TreeExplainer baru untuk model CatBoost v3.")
        _explainer_cache[key] = shap.TreeExplainer(model)
    return _explainer_cache[key]


def get_shap_reasons(
    pipeline,
    features_df: pd.DataFrame,
    top_n: int = TOP_N_REASONS,
    min_contribution_pct: float = MIN_CONTRIBUTION_PCT,
) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(features_df)

    explainer = _get_explainer(model)
    shap_values = explainer.shap_values(X_transformed)
    row_shap = shap_values[0] if getattr(shap_values, "ndim", 1) == 2 else shap_values

    if hasattr(X_transformed, "columns"):
        feature_names = list(X_transformed.columns)
        row_values = X_transformed.iloc[0]
    else:
        feature_names = list(features_df.columns)
        row_values = features_df.iloc[0]

    total_abs = sum(abs(v) for v in row_shap) or 1.0

    contributions = []
    for name, value in zip(feature_names, row_shap):
        if name in EXCLUDED_FROM_DISPLAY:
            continue

        pct = (float(value) / total_abs) * 100
        if abs(pct) < min_contribution_pct:
            continue

        raw_value = row_values[name] if name in row_values.index else None

        contributions.append({
            "feature": name,
            "label": get_feature_label(name),
            "value_display": _format_value(name, raw_value),
            "direction": "menaikkan_risiko" if value > 0 else "menurunkan_risiko",
            "contribution_pct": round(pct, 1),
        })

    contributions.sort(key=lambda x: abs(x["contribution_pct"]), reverse=True)
    return contributions[:top_n]