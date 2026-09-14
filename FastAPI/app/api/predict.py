# Bagian 1 — Import dan setup router. APIRouter dengan prefix="/predict"
#  berarti path lengkap nanti jadi /predict/existing dan /predict/new — tidak
#  perlu tulis /predict berulang di setiap endpoint.
# FastAPI/app/api/predict.py
import logging

from fastapi import APIRouter, Response

from app.schemas.applicant import (
    ApplicantModeExisting,
    NewApplicantRequest,
    PredictionResponse,
)
from app.services.prediction_service import predict_existing, predict_new
from app.services.pdf_service import build_prediction_pdf

logger = logging.getLogger("home_credit_api")
router = APIRouter(prefix="/predict", tags=["prediction"])


# Bagian 2 — Endpoint Mode Existing. Router ini sengaja tipis — tidak ada
# try/except di sini. Kalau sk_id_curr tidak ditemukan, predict_existing()
# akan melempar SKIDNotFoundError, dan itu otomatis ditangkap oleh handler
# yang sudah didaftarkan di exceptions.py lewat register_exception_handlers()
# — bukan ditangani manual di sini. Ini yang membuat pola service layer efektif:
# router cuma jadi jembatan HTTP, logic sesungguhnya ada di services/.
@router.post("/existing", response_model=PredictionResponse, summary="Prediksi nasabah existing (lookup histori)")
async def predict_existing_applicant(payload: ApplicantModeExisting) -> PredictionResponse:
    logger.info(f"Request Mode Existing: SK_ID_CURR={payload.sk_id_curr}")
    return predict_existing(payload.sk_id_curr)


# Bagian 3 — Endpoint Mode Baru. Menerima NewApplicantRequest yang membungkus
# applicant + context, lalu meneruskan keduanya ke predict_new(). Kalau data pemohon
# melanggar validator (usia tidak wajar, dst.), FastAPI otomatis mengembalikan 422
# sebelum kode ini sempat jalan — itu keuntungan validasi Pydantic yang sudah kita
# bahas di awal soal alasan memilih FastAPI.
@router.post("/new", response_model=PredictionResponse, summary="Prediksi nasabah baru (form manual)")
async def predict_new_applicant(payload: NewApplicantRequest) -> PredictionResponse:
    logger.info(f"Request Mode Baru: {payload.applicant.name_income_type}, status={payload.applicant.status_pekerjaan}")
    return predict_new(payload.applicant, payload.context)


# Bagian 4 — Endpoint PDF Mode Existing. Sengaja memanggil ULANG predict_existing()
# (bukan menerima hasil prediksi dari client) — ini keputusan arsitektur "Opsi A"
# yang disepakati: PDF harus otentik, dihitung ulang oleh backend, bukan
# dipercaya begitu saja dari data yang dikirim balik oleh browser (rawan
# dimanipulasi lewat DevTools sebelum dikirim).
@router.post("/existing/report", summary="Unduh PDF hasil evaluasi nasabah existing")
async def predict_existing_report(payload: ApplicantModeExisting) -> Response:
    logger.info(f"Request PDF Mode Existing: SK_ID_CURR={payload.sk_id_curr}")
    result = predict_existing(payload.sk_id_curr)

    pdf_bytes = build_prediction_pdf(
        mode="existing",
        result=result.model_dump(),
        sk_id_curr=payload.sk_id_curr,
    )
    filename = f"laporan-evaluasi-{payload.sk_id_curr}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# Bagian 5 — Endpoint PDF Mode Baru. Sama prinsipnya: recompute penuh lewat
# predict_new(), termasuk re-validasi Pydantic pada payload.applicant.
@router.post("/new/report", summary="Unduh PDF hasil evaluasi nasabah baru")
async def predict_new_report(payload: NewApplicantRequest) -> Response:
    logger.info(f"Request PDF Mode Baru: {payload.applicant.name_income_type}, status={payload.applicant.status_pekerjaan}")
    result = predict_new(payload.applicant, payload.context)

    pdf_bytes = build_prediction_pdf(
        mode="new",
        result=result.model_dump(),
        sk_id_curr=None,
    )
    filename = "laporan-evaluasi-nasabah-baru.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )