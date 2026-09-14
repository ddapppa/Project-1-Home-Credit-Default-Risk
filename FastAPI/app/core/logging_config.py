# Bagian 1 — Import dan fungsi utama setup_logging(). Fungsi ini dipanggil sekali saat aplikasi 
# startup (di main.py), mengatur format log dan level-nya berdasarkan .env.
# FastAPI/app/core/logging_config.py
import logging
import sys
from pathlib import Path

from app.core.config import get_settings, BASE_DIR


def setup_logging() -> None:
    settings = get_settings()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Bagian 2 — Console handler (selalu aktif). Ini yang menampilkan log ke terminal saat testing lokal 
# dengan uvicorn, dan yang ditangkap otomatis oleh Vercel di production.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

# Bagian 3 — File handler (hanya lokal). Kalau environment == "local", tambahkan juga log ke file 
# supaya kamu punya riwayat debugging tersimpan di laptop. Folder logs/ dibuat otomatis kalau belum ada.
    handlers = [console_handler]

    if settings.environment == "local":
        log_dir = BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

# Bagian 4 — Terapkan konfigurasi ke logger utama aplikasi. Nama logger "home_credit_api" ini 
# harus sama persis dengan yang dipakai di exceptions.py (logging.getLogger("home_credit_api")) — 
# kalau namanya beda, Python akan menganggapnya logger yang berbeda dan konfigurasi ini tidak akan 
# berlaku untuknya.
    logger = logging.getLogger("home_credit_api")
    logger.setLevel(settings.log_level.upper())
    logger.handlers.clear()  # hindari duplikat handler kalau setup_logging() terpanggil >1x
    for handler in handlers:
        logger.addHandler(handler)

    logger.propagate = False  # cegah log dobel ke root logger

# Bagian 5 — Redam log library pihak ketiga yang terlalu berisik. uvicorn dan catboost kadang 
# mengeluarkan log INFO/DEBUG yang tidak relevan untuk debugging aplikasi kita — ini opsional tapi 
# bikin output terminal lebih bersih.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)