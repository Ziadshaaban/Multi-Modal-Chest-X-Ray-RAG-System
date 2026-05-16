import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# --- Auto-load .env so credentials (GOOGLE_API_KEY, KAGGLE_*, HF_TOKEN, etc.)
# --- are available without a separate PowerShell command each session.
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# --- Cache redirects: keep large caches off the C: drive. Must happen BEFORE
# --- importing torch / transformers / kagglehub so they pick these up.
CACHE_ROOT = PROJECT_ROOT.parent / "cache"
HF_CACHE = CACHE_ROOT / "huggingface"
KAGGLEHUB_CACHE = CACHE_ROOT / "kagglehub"
PIP_CACHE = CACHE_ROOT / "pip"
for _d in [HF_CACHE, KAGGLEHUB_CACHE, PIP_CACHE]:
    _d.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))
# kagglehub uses different env var names across versions - set all known variants
os.environ.setdefault("KAGGLEHUB_CACHE", str(KAGGLEHUB_CACHE))
os.environ.setdefault("KAGGLEHUB_CACHE_FOLDER", str(KAGGLEHUB_CACHE))
os.environ.setdefault("PIP_CACHE_DIR", str(PIP_CACHE))

import torch  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
QA_DATA_DIR = DATA_DIR / "qa"
RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison_tables"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, QA_DATA_DIR, COMPARISON_DIR, FIGURES_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
if torch.cuda.is_available():
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
    CUDA_DEVICE_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
else:
    CUDA_DEVICE_NAME = "CPU"
    CUDA_DEVICE_MEMORY = 0.0

print(f"[CONFIG] Device: {DEVICE} | GPUs: {CUDA_DEVICE_COUNT} | Device Name: {CUDA_DEVICE_NAME} | VRAM: {CUDA_DEVICE_MEMORY:.1f}GB")
print(f"[CONFIG] HF cache:        {os.environ['HF_HOME']}")
print(f"[CONFIG] Kagglehub cache: {os.environ['KAGGLEHUB_CACHE']}")

MODEL_IDS = {
    "medgemma_4b": "google/medgemma-4b-it",
    "colpali": "vidore/colpali-v1.3",
    "clip": "openai/clip-vit-base-patch32",
}

MEDGEMMA_BACKEND = os.getenv("MEDGEMMA_BACKEND", "local")
MEDGEMMA_USE_4BIT = CUDA_DEVICE_MEMORY < 8.0 if CUDA_DEVICE_MEMORY > 0 else False

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
RETRIEVAL_TOP_K_REPORT = int(os.getenv("RETRIEVAL_TOP_K_REPORT", "3"))

DATASET_SUBSET_SIZE = int(os.getenv("DATASET_SUBSET_SIZE", "1000"))
TRAIN_CORPUS_SIZE = int(os.getenv("TRAIN_CORPUS_SIZE", "750"))
EVAL_SET_SIZE = int(os.getenv("EVAL_SET_SIZE", "150"))

MAX_REPORT_LENGTH = 1024
MAX_QUESTION_LENGTH = 256
MAX_ANSWER_LENGTH = 512

QA_GENERATION_MODEL = os.getenv("QA_GENERATION_MODEL", "gemini")

LLM_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

print(f"[CONFIG] MedGemma Backend: {MEDGEMMA_BACKEND} | Use 4-bit: {MEDGEMMA_USE_4BIT}")
print(f"[CONFIG] Retrieval K: {RETRIEVAL_K} | Report K: {RETRIEVAL_TOP_K_REPORT}")
print(f"[CONFIG] Dataset subset size: {DATASET_SUBSET_SIZE} | Train: {TRAIN_CORPUS_SIZE} | Eval: {EVAL_SET_SIZE}")
