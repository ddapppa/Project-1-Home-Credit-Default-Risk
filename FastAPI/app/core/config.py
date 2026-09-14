# FastAPI/app/core/config.py
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Folder src/ (root repo, sejajar dengan FastAPI/) diinstall sebagai package Python
# via `pip install -e .` dari root repo (lihat pyproject.toml di root repo).
# Ini menggantikan pendekatan sys.path manual yang sebelumnya hanya bekerja lokal.
# Untuk Vercel: Root Directory project WAJIB diarahkan ke ROOT REPO (bukan FastAPI/),
# supaya pyproject.toml dan src/ ikut ter-deploy dan package terinstall saat build.


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- App Metadata ---
    app_name: str = "Home Credit Default Risk API"
    app_version: str = "1.0.0"
    environment: str = "local"  # "local" | "production"

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True

    # --- Path model & data (relatif terhadap BASE_DIR) ---
    model_path: str = "app/ml/catboost_v3_pipeline.joblib"
    demo_lookup_path: str = "app/data/demo_lookup.csv.gz"
    feature_metadata_path: str = "app/ml/feature_engineering_metadata_v3.json"  # <- BARU

    # --- Threshold keputusan (freeze dari tahap Evaluation) ---
    decision_threshold: float = 0.6669

    # --- Logging ---
    log_level: str = "INFO"

    # --- CORS ---
    allowed_origins: str = "http://127.0.0.1:8000,http://localhost:8000"

    @property
    def resolved_model_path(self) -> Path:
        return BASE_DIR / self.model_path

    @property
    def resolved_demo_lookup_path(self) -> Path:
        return BASE_DIR / self.demo_lookup_path

    @property
    def resolved_feature_metadata_path(self) -> Path:  # <- BARU
        return BASE_DIR / self.feature_metadata_path

    @property
    def origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    def model_post_init(self, __context) -> None:
        if self.environment == "production" and self.debug:
            self.debug = False


@lru_cache
def get_settings() -> Settings:
    return Settings()