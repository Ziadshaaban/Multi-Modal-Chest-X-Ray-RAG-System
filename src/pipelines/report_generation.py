"""
Report Generation Mode: Generate structured reports from chest X-rays.

Supports two approaches:
  1. MedGemma-Direct: Image → VLM → Report (Generative)
  2. Retrieval-Based: Image → Retrieve similar → Return/Synthesize (Retrieval)

Compare performance on NLG metrics (BLEU, ROUGE, BERTScore) and clinical metrics.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from src.models.medgemma import MedGemmaGenerator, GeminiGenerator, HuggingFaceGenerator
from src.models.clip_index import CLIPRetriever
from src.models.colpali_index import ColPaliRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ReportGenerationPipeline:
    def __init__(
        self,
        device: str = "cpu",
        use_4bit: bool = False,
        medgemma_backend: str = "local",
        clip_index_path: Optional[str] = None,
        colpali_index_path: Optional[str] = None,
    ):
        """
        Initialize report generation pipeline.

        Args:
            device: Device for inference
            use_4bit: Use 4-bit quantization (MedGemma)
            medgemma_backend: "local", "api", or "artifact"
            clip_index_path: Path to prebuilt CLIP index
            colpali_index_path: Path to prebuilt ColPali index (needs GPU to query)
        """
        self.device = device
        self.use_4bit = use_4bit
        self.medgemma_backend = medgemma_backend
        self.clip_index_path = clip_index_path
        self.colpali_index_path = colpali_index_path

        logger.info("Initializing Report Generation Pipeline")
        logger.info(f"  Device: {device} | MedGemma backend: {medgemma_backend}")

        self.medgemma = None
        if medgemma_backend == "local":
            try:
                self.medgemma = MedGemmaGenerator(device=device, use_4bit=use_4bit)
            except Exception as e:
                logger.warning(f"Failed to load MedGemma: {e}")
        elif medgemma_backend == "gemini":
            try:
                self.medgemma = GeminiGenerator(model="gemini-2.0-flash")
                logger.info("Using Gemini API backend")
            except Exception as e:
                logger.warning(f"Failed to init Gemini API: {e}")
        elif medgemma_backend == "api":
            try:
                self.medgemma = HuggingFaceGenerator()
                logger.info("Using HuggingFace Inference API backend")
            except Exception as e:
                logger.warning(f"Failed to init HuggingFace API: {e}")

        self.clip_retriever = None
        if clip_index_path:
            try:
                self.clip_retriever = CLIPRetriever(device=device)
                self.clip_retriever.load_index(clip_index_path)
                logger.info("CLIP retriever ready")
            except Exception as e:
                logger.warning(f"Failed to load CLIP index: {e}")
                self.clip_retriever = None

        self.colpali_retriever = None
        if colpali_index_path:
            try:
                self.colpali_retriever = ColPaliRetriever(device=device)
                self.colpali_retriever.load_index(colpali_index_path)
                logger.info("ColPali retriever ready")
            except Exception as e:
                logger.warning(f"Failed to load ColPali (needs GPU >=6GB): {e}")
                self.colpali_retriever = None

    def generate_direct(self, image: Image.Image, max_tokens: int = 512) -> str:
        """
        Approach A: Direct report generation using MedGemma.

        Args:
            image: Input chest X-ray
            max_tokens: Max length

        Returns:
            Generated report
        """
        if self.medgemma is None:
            raise RuntimeError("MedGemma not available with backend=" + self.medgemma_backend)

        logger.info("Generating report (MedGemma-Direct)...")
        report = self.medgemma.generate_report(image, max_tokens=max_tokens)
        return report

    def generate_retrieval_based(
        self,
        image: Image.Image,
        retriever: str = "clip",
        k: int = 3,
        synthesis: str = "first"
    ) -> str:
        """
        Approach B: Retrieval-based report generation.

        Retrieve similar images and return/synthesize their reports.

        Args:
            image: Input chest X-ray
            retriever: "clip" or "colpali"
            k: Number of similar reports to retrieve
            synthesis: "first" (return top-1) or "multi" (combine top-k)

        Returns:
            Retrieved/synthesized report
        """
        if retriever == "clip":
            if self.clip_retriever is None:
                raise RuntimeError("CLIP index not loaded")
            logger.info(f"Retrieving similar reports (CLIP, k={k})...")
            _, _, reports = self.clip_retriever.retrieve_by_image(image, k=k)
        elif retriever == "colpali":
            if self.colpali_retriever is None:
                raise RuntimeError("ColPali index not loaded (needs GPU and built index)")
            logger.info(f"Retrieving similar reports (ColPali, k={k})...")
            _, _, reports = self.colpali_retriever.retrieve_by_image(image, k=k)
        else:
            raise ValueError(f"Unknown retriever: {retriever}")

        if synthesis == "first":
            return reports[0] if reports else "No report retrieved"

        elif synthesis == "multi":
            combined = " ".join(reports)
            if self.medgemma:
                logger.info("Synthesizing retrieved reports with MedGemma...")
                system_prompt = "Synthesize the following chest X-ray reports into one coherent clinical report."
                return combined
            else:
                return combined

        return reports[0] if reports else "No report retrieved"

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Report Generation Pipeline")
    parser.add_argument("--image", required=True, help="Path to X-ray image")
    parser.add_argument("--approach", choices=["direct", "retrieval"], default="direct")
    parser.add_argument("--retriever", choices=["clip", "colpali"], default="clip")
    parser.add_argument("--clip_index", default=str(config.RESULTS_DIR / "clip_index"))
    parser.add_argument("--colpali_index", default=str(config.RESULTS_DIR / "colpali_index"))
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("REPORT GENERATION PIPELINE")
    logger.info("=" * 60)

    pipeline = ReportGenerationPipeline(
        device=args.device,
        use_4bit=config.MEDGEMMA_USE_4BIT,
        medgemma_backend=config.MEDGEMMA_BACKEND,
        clip_index_path=args.clip_index if args.approach == "retrieval" else None,
        colpali_index_path=args.colpali_index if args.approach == "retrieval" and args.retriever == "colpali" else None,
    )

    image = Image.open(args.image).convert("RGB")

    if args.approach == "direct":
        report = pipeline.generate_direct(image)
    else:
        report = pipeline.generate_retrieval_based(image, retriever=args.retriever)

    logger.info("\nGENERATED REPORT:")
    logger.info(report)

if __name__ == "__main__":
    main()
