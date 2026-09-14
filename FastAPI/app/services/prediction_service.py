# FastAPI/app/services/prediction_service.py
import logging
from functools import lru_cache

import joblib
import pandas as pd

from app.core.config import get_settings
# "src" adalah package Python yang terinstall via `pip install -e .` dari root repo
# (lihat pyproject.toml di root). Import ini bekerja identik di semua environment
# tanpa perlu sys.path manual, karena Python mengenali "src" seperti library biasa.
from src.modeling.train import CatBoostColumnSelector
from app.core.exceptions import ModelNotLoadedError

from app.core.feature_lookup import build_new_applicant_features, get_existing_applicant_features
from app.schemas.applicant import ApplicantModeNew, ApplicationContext, PredictionResponse
from app.services.explainability_service import get_shap_reasons

logger = logging.getLogger("home_credit_api")
settings = get_settings()

@lru_cache
def load_pipeline():
    try:
        pipeline = joblib.load(settings.resolved_model_path)
        logger.info(f"Pipeline CatBoost v3 berhasil dimuat dari {settings.resolved_model_path}")
        return pipeline
    except Exception as e:
        raise ModelNotLoadedError(f"Gagal load {settings.resolved_model_path}: {e}")

# Bagian 2 — Business rule overlay: margin review untuk nasabah baru. Ini implementasi konkret dari 
# diskusi kita soal cold-start problem: karena nasabah baru kehilangan sinyal terkuat 
# (EXT_SOURCE_1/2/3 dan histori bureau), skornya secara inheren kurang bisa dipercaya sepenuhnya. 
# Kalau probabilitasnya jatuh dekat threshold (bukan jauh di atas/bawah), kita paksa jadi status review manual, bukan auto-approve.
NEW_APPLICANT_REVIEW_MARGIN = 0.05  # ambang keraguan di sekitar threshold, khusus Mode Baru
HIGH_CREDIT_INCOME_RATIO = 5.0      # heuristik bisnis: kredit > 5x income -> butuh verifikasi tambahan


def _apply_business_overlay(
    probability: float, threshold: float, mode: str, applicant: ApplicantModeNew | None
) -> str:
    if probability >= threshold:
        return "REVIEW/REJECT"

    if mode == "new" and applicant is not None:
        near_threshold = (threshold - probability) <= NEW_APPLICANT_REVIEW_MARGIN
        credit_income_ratio = applicant.amt_credit / applicant.amt_income_total
        high_value_loan = credit_income_ratio >= HIGH_CREDIT_INCOME_RATIO

        if near_threshold or high_value_loan:
            logger.info(
                f"Overlay aktif: near_threshold={near_threshold}, "
                f"high_value_loan={high_value_loan} (ratio={credit_income_ratio:.2f})"
            )
            return "REVIEW/REJECT"

    return "APPROVE"


# Bagian 3 — Fungsi prediksi Mode Existing. Langsung lookup histori, tidak perlu overlay tambahan 
# karena datanya lengkap (termasuk EXT_SOURCE), sesuai desain kita bahwa Mode Existing merepresentasikan kondisi ideal.
def predict_existing(sk_id_curr: int) -> PredictionResponse:
    pipeline = load_pipeline()
    X = get_existing_applicant_features(sk_id_curr)

    probability = float(pipeline.predict_proba(X)[:, 1][0])
    threshold = settings.decision_threshold
    keputusan = "REVIEW/REJECT" if probability >= threshold else "APPROVE"

    logger.info(f"[existing] SK_ID_CURR={sk_id_curr} proba={probability:.4f} keputusan={keputusan}")

    try:
        reasons = get_shap_reasons(pipeline, X)
    except Exception as e:
        logger.warning(f"Gagal menghitung SHAP reasons untuk SK_ID_CURR={sk_id_curr}: {e}")
        reasons = None

    return PredictionResponse(
        mode="existing",
        probability_default=probability,
        threshold=threshold,
        keputusan=keputusan,
        disclaimer=None,
        reasons=reasons,
        context=None,
    )

# Bagian 4 — Fungsi prediksi Mode Baru. Menyertakan overlay dari Bagian 2, dan meneruskan
# context (UI-only) ke response tanpa pernah menyentuh model.
def predict_new(
    applicant: ApplicantModeNew, context: ApplicationContext | None = None
) -> PredictionResponse:
    pipeline = load_pipeline()
    X = build_new_applicant_features(applicant)

    probability = float(pipeline.predict_proba(X)[:, 1][0])
    threshold = settings.decision_threshold
    keputusan = _apply_business_overlay(probability, threshold, mode="new", applicant=applicant)

    logger.info(f"[new] proba={probability:.4f} keputusan={keputusan}")

    return PredictionResponse(
        mode="new",
        probability_default=probability,
        threshold=threshold,
        keputusan=keputusan,
        context=context,
    )

NEW_APPLICANT_DISCLAIMER = (
    "PERINGATAN: Kombinasi data nasabah baru ini (tanpa histori kredit apa pun DAN tanpa skor "
    "eksternal EXT_SOURCE sama sekali) telah divalidasi TIDAK PERNAH ditemukan pada 307.511 data "
    "training model (0 dari 307.511 baris memiliki kombinasi identik). Probabilitas yang "
    "ditampilkan adalah hasil ekstrapolasi model ke wilayah data yang belum pernah dipelajari, "
    "sehingga TIDAK dijadikan dasar keputusan otomatis. Sistem mewajibkan verifikasi manual penuh "
    "(dokumen usaha, referensi pihak ketiga, survei lapangan) untuk semua nasabah baru, terlepas "
    "dari nilai probabilitas yang muncul."
)


def predict_new(
    applicant: ApplicantModeNew, context: ApplicationContext | None = None
) -> PredictionResponse:
    pipeline = load_pipeline()
    X = build_new_applicant_features(applicant)

    probability = float(pipeline.predict_proba(X)[:, 1][0])
    threshold = settings.decision_threshold

    # Mode Baru SELALU out-of-distribution (EXT_SOURCE + histori NaN sekaligus,
    # tervalidasi 0/307.511 baris training). Keputusan dikunci REVIEW/REJECT,
    # tidak pernah auto-APPROVE, terlepas dari nilai probability_default.
    keputusan = "REVIEW/REJECT"

    logger.info(
        f"[new] proba={probability:.4f} keputusan={keputusan} "
        f"(forced review — input out-of-distribution, lihat NEW_APPLICANT_DISCLAIMER)"
    )

    return PredictionResponse(
        mode="new",
        probability_default=probability,
        threshold=threshold,
        keputusan=keputusan,
        disclaimer=NEW_APPLICANT_DISCLAIMER,
        context=context,
    )