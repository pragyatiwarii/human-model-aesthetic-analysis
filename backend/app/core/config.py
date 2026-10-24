from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"
RUNTIME_DIR = DATA_DIR / "runtime"
AGIQA_DIR = DATA_DIR / "agiqa3k"


class Settings(BaseSettings):
    app_name: str = "Human-Model Aesthetic Analysis"
    api_prefix: str = "/api/v1"
    frontend_origin: str = "http://127.0.0.1:5173"

    enable_ml_models: bool = True
    yolo_model: str = "yolov8n-seg.pt"
    detection_confidence: float = 0.55

    enable_context_vlm: bool = True
    context_vlm_model: str = "Salesforce/blip-image-captioning-base"
    context_vlm_device: str = "auto"
    context_vlm_max_new_tokens: int = 40

    agiqa_metadata_path: Path = AGIQA_DIR / "data.csv"
    agiqa_images_dir: Path = AGIQA_DIR / "images"

    uploads_dir: Path = RUNTIME_DIR / "uploads"
    annotated_dir: Path = RUNTIME_DIR / "annotated"
    regions_dir: Path = RUNTIME_DIR / "regions"
    counterfactuals_dir: Path = RUNTIME_DIR / "counterfactuals"
    aesthetic_model_path: Path = RUNTIME_DIR / "models" / "aesthetic_ridge_v1.json"

    aesthetic_train_limit: int = 1000
    aesthetic_ridge_lambda: float = 3.0

    # Extend the clean v1.4 Module 1 database so existing image/region IDs remain valid.
    db_path: Path = RUNTIME_DIR / "module1_v14.db"

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

for directory in (
    DATA_DIR,
    RUNTIME_DIR,
    settings.agiqa_images_dir,
    settings.uploads_dir,
    settings.annotated_dir,
    settings.regions_dir,
    settings.counterfactuals_dir,
    settings.aesthetic_model_path.parent,
):
    directory.mkdir(parents=True, exist_ok=True)
