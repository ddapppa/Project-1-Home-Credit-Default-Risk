# Bagian 1 — Base exception class. Semua custom exception project ini mewarisi satu class dasar 
# AppBaseException, supaya nanti gampang di-except secara umum (misal di logging) dan setiap exception 
# otomatis punya message yang konsisten.
# FastAPI/app/core/exceptions.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("home_credit_api")


class AppBaseException(Exception):
    """Base class untuk semua custom exception di project ini."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

# Bagian 2 — SKIDNotFoundError. Dilempar oleh feature_lookup.py saat Mode Existing dipakai tapi 
# sk_id_curr yang diinput user tidak ada di demo_lookup.parquet. Ini beda dari error validasi Pydantic 
# biasa karena masalahnya bukan "format data salah", tapi "data valid secara format, tapi tidak ditemukan" — 
# secara HTTP semantik ini seharusnya 404 Not Found, bukan 422 Unprocessable Entity.
class SKIDNotFoundError(AppBaseException):
    """SK_ID_CURR yang diinput tidak ditemukan di demo_lookup.parquet (Mode Existing)."""
    def __init__(self, sk_id_curr: int):
        self.sk_id_curr = sk_id_curr
        super().__init__(
            f"SK_ID_CURR {sk_id_curr} tidak ditemukan di data histori demo. "
            f"Gunakan Mode Nasabah Baru jika ini benar-benar nasabah baru."
        )

# Bagian 3 — InvalidApplicantDataError. Ini untuk aturan bisnis yang tidak bisa divalidasi Pydantic murni 
# (Pydantic hanya cek tipe data & format per-field). Contoh kasus nyata dari desain kita kemarin: kombinasi 
# status_pekerjaan="Bekerja" tapi tanggal_mulai_kerja lebih baru dari tanggal_lahir + 15 tahun 
# (secara logis mustahil) — itu aturan lintas-field yang lebih cocok dicek di service layer, bukan schema.
class InvalidApplicantDataError(AppBaseException):
    """Data pemohon valid secara format, tapi melanggar aturan bisnis (cross-field validation)."""
    def __init__(self, detail: str):
        super().__init__(f"Data pemohon tidak valid: {detail}")

# Bagian 4 — ModelNotLoadedError. Dilempar kalau main.py gagal me-load catboost_v3_pipeline.joblib saat 
# startup (misal file tidak ketemu, atau versi library tidak cocok seperti yang kita tes sebelumnya). 
# Ini kondisi fatal — server sebaiknya tidak menerima request sama sekali kalau model belum siap, jadi 
# status-nya 503 Service Unavailable.
class ModelNotLoadedError(AppBaseException):
    """Pipeline model gagal dimuat saat startup aplikasi."""
    def __init__(self, detail: str):
        super().__init__(f"Model belum siap: {detail}")

# Bagian 5 — Exception handler. Ini fungsi-fungsi yang nanti didaftarkan ke instance FastAPI() 
# di main.py lewat app.add_exception_handler(...). Setiap handler menangkap satu jenis exception 
# dan mengubahnya jadi JSONResponse dengan status code dan format body yang konsisten 
# ({"error": "...", "detail": "..."}), supaya frontend JS-mu bisa menangani error dengan cara 
# yang seragam, apa pun jenis error-nya.
async def sk_id_not_found_handler(request: Request, exc: SKIDNotFoundError) -> JSONResponse:
    logger.warning(f"SKIDNotFoundError: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "sk_id_not_found", "detail": exc.message},
    )


async def invalid_applicant_data_handler(request: Request, exc: InvalidApplicantDataError) -> JSONResponse:
    logger.warning(f"InvalidApplicantDataError: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "invalid_applicant_data", "detail": exc.message},
    )


async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    logger.error(f"ModelNotLoadedError: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "model_not_loaded", "detail": exc.message},
    )

# Bagian 6 — Fungsi pendaftaran terpusat. Daripada main.py menulis tiga baris app.add_exception_handler(...) 
# satu-satu (dan berisiko lupa kalau nanti nambah exception baru), semua didaftarkan lewat satu fungsi 
# register_exception_handlers(app) yang tinggal dipanggil sekali di main.py.
def register_exception_handlers(app) -> None:
    app.add_exception_handler(SKIDNotFoundError, sk_id_not_found_handler)
    app.add_exception_handler(InvalidApplicantDataError, invalid_applicant_data_handler)
    app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
