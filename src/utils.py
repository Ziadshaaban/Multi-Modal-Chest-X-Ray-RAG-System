import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image(image_path: str) -> Image.Image:
    """Load and validate image."""
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise

def save_json(data: Dict[str, Any], output_path: str) -> None:
    """Save data to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved to {output_path}")

def load_json(json_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def clean_report_text(text: str) -> str:
    """Basic text cleaning for medical reports."""
    if not text:
        return ""
    text = text.strip()
    text = " ".join(text.split())
    return text

def parse_report_sections(report: str) -> Dict[str, str]:
    """Try to parse FINDINGS and IMPRESSION sections."""
    sections = {"findings": "", "impression": "", "full": report}

    report_lower = report.lower()

    if "findings" in report_lower:
        findings_idx = report_lower.index("findings")
        findings_start = report.find(":", findings_idx) + 1 if ":" in report[findings_idx:] else findings_idx + 8

        impression_idx = report_lower.find("impression")
        if impression_idx > findings_idx:
            sections["findings"] = report[findings_start:impression_idx].strip()
            impression_start = report.find(":", impression_idx) + 1 if ":" in report[impression_idx:] else impression_idx + 11
            sections["impression"] = report[impression_start:].strip()
        else:
            sections["findings"] = report[findings_start:].strip()
    elif "impression" in report_lower:
        impression_idx = report_lower.index("impression")
        impression_start = report.find(":", impression_idx) + 1 if ":" in report[impression_idx:] else impression_idx + 11
        sections["impression"] = report[impression_start:].strip()

    if not sections["findings"] and not sections["impression"]:
        sections["findings"] = report

    return sections

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }

def print_config_info(config) -> None:
    """Print configuration info."""
    logger.info("=" * 60)
    logger.info("SYSTEM CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"CUDA Available: {config.DEVICE == 'cuda'}")
    logger.info(f"GPU VRAM: {config.CUDA_DEVICE_MEMORY:.1f} GB")
    logger.info(f"MedGemma Backend: {config.MEDGEMMA_BACKEND}")
    logger.info(f"Use 4-bit Quantization: {config.MEDGEMMA_USE_4BIT}")
    logger.info("=" * 60)
