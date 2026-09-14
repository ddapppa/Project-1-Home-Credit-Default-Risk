#  Import semua layer yang sudah dibuat.
# FastAPI/app/main.py
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.core.exceptions import register_exception_handlers, ModelNotLoadedError
from app.core.feature_lookup import load_tree_feature_columns
from app.services.prediction_service import load_pipeline
from app.api.predict import router as predict_router

settings = get_settings()

# Bagian 2 — lifespan: apa yang terjadi saat startup dan shutdown. Ini bagian paling
# penting untuk keandalan aplikasi: pipeline model dan metadata fitur dipaksa dimuat 
# saat startup, bukan ditunda sampai request pertama datang. Kalau ada yang gagal 
# (file tidak ketemu, versi library tidak cocok), aplikasi akan gagal start dengan 
# error yang jelas di log — jauh lebih baik daripada error muncul mendadak saat user 
# pertama mencoba /predict.
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger = logging.getLogger("home_credit_api")
    logger.info(f"Starting {settings.app_name} v{settings.app_version} ({settings.environment})")

    try:
        load_pipeline()
        load_tree_feature_columns()
        logger.info("Pipeline dan metadata fitur siap. Aplikasi ready menerima request.")
    except ModelNotLoadedError as e:
        logger.error(f"STARTUP GAGAL: {e.message}")
        raise

    yield  # <- aplikasi berjalan di sini, menangani request

    logger.info("Aplikasi shutting down.")

# Bagian 3 — Buat instance FastAPI() dengan lifespan terpasang. Parameter title/
# version otomatis muncul di dokumentasi Swagger UI (/docs) yang di-generate FastAPI —
#  ini salah satu keunggulan yang kita bahas di awal soal alasan pemilihan framework ini.
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# Bagian 4 — CORS middleware. Mengizinkan frontend (kalau nanti dipisah domain) 
# memanggil API ini. Daftar origin diambil dari .env lewat settings.origins_list 
# yang sudah kita siapkan di config.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bagian 5 — Daftarkan exception handler dan router. Satu baris untuk masing-masing,
# memanfaatkan fungsi terpusat yang sudah kita buat di exceptions.py.
register_exception_handlers(app)
app.include_router(predict_router)

#Bagian 6 — Mount static/ dan setup templates/. static/ untuk CSS/JS custom-mu, 
# templates/ untuk file HTML yang di-render Jinja2. Path dihitung relatif dari lokasi 
# main.py sendiri, konsisten dengan pola BASE_DIR di config.py.
APP_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


# Bagian 7 — Endpoint dasar: halaman form dan health check. GET /
# merender halaman utama (form input, dikerjakan di tahap berikutnya). GET /health
# dipakai untuk cek cepat apakah server hidup — berguna nanti kalau di-deploy ke
# Vercel untuk memastikan function tidak "cold" bermasalah.
@app.get("/", tags=["frontend"])
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "app_name": settings.app_name, "threshold": settings.decision_threshold},
    )


@app.get("/health", tags=["monitoring"])
async def health_check():
    return {"status": "ok", "environment": settings.environment, "app_version": settings.app_version}

