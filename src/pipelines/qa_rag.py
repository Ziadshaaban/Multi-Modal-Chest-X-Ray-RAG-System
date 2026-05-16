"""
QA Mode (RAG): Retrieval-Augmented Question Answering over chest X-rays.

Flow:
  1. Index corpus reports with retriever (CLIP or ColPali)
  2. Embed query image, retrieve top-k similar reports as context
  3. Feed image + question + context to MedGemma → grounded answer
  4. Ablate: with/without context to show RAG value

Compare retrievers: CLIP (local) vs ColPali (cloud).
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

class QARagPipeline:
    def __init__(
        self,
        device: str = "cpu",
        use_4bit: bool = False,
        clip_index_path: Optional[str] = None,
        colpali_index_path: Optional[str] = None,
        medgemma_backend: str = "local"
    ):
        """
        Initialize QA RAG pipeline.

        Args:
            device: Device for inference
            use_4bit: Use 4-bit quantization
            clip_index_path: Path to prebuilt CLIP index
            colpali_index_path: Path to prebuilt ColPali index
            medgemma_backend: "local", "api", or "artifact"
        """
        self.device = device
        self.use_4bit = use_4bit
        self.medgemma_backend = medgemma_backend

        logger.info("Initializing QA RAG Pipeline")

        if medgemma_backend == "local":
            try:
                self.medgemma = MedGemmaGenerator(device=device, use_4bit=use_4bit)
            except Exception as e:
                logger.warning(f"Failed to load MedGemma: {e}")
                self.medgemma = None
        elif medgemma_backend == "gemini":
            try:
                self.medgemma = GeminiGenerator(model="gemini-2.0-flash")
                logger.info("Using Gemini API backend")
            except Exception as e:
                logger.warning(f"Failed to init Gemini API: {e}")
                self.medgemma = None
        elif medgemma_backend == "api":
            try:
                self.medgemma = HuggingFaceGenerator()
                logger.info("Using HuggingFace Inference API backend")
            except Exception as e:
                logger.warning(f"Failed to init HuggingFace API: {e}")
                self.medgemma = None
        else:
            self.medgemma = None

        self.clip_retriever = None
        if clip_index_path:
            self.clip_retriever = CLIPRetriever(device=device)
            try:
                self.clip_retriever.load_index(clip_index_path)
                logger.info("✓ CLIP index loaded")
            except Exception as e:
                logger.warning(f"Failed to load CLIP index: {e}")

        self.colpali_retriever = None
        if colpali_index_path:
            try:
                self.colpali_retriever = ColPaliRetriever(device=device)
                self.colpali_retriever.load_index(colpali_index_path)
                logger.info("ColPali retriever ready")
            except Exception as e:
                logger.warning(f"Failed to load ColPali (needs GPU >=6GB): {e}")
                self.colpali_retriever = None

    def answer_question(
        self,
        image: Image.Image,
        question: str,
        retriever: str = "clip",
        use_context: bool = True,
        k: int = 3,
        max_tokens: int = 256
    ) -> Dict[str, str]:
        """
        Answer a question about an image using RAG.

        Args:
            image: Input chest X-ray
            question: Clinical question
            retriever: "clip" or "colpali"
            use_context: Include retrieved context in generation
            k: Number of context documents to retrieve
            max_tokens: Max length of answer

        Returns:
            Dict with keys: answer, context, retriever, question
        """
        context = None
        retrieved_reports = []

        if use_context:
            if retriever == "clip":
                if self.clip_retriever is None:
                    logger.warning("CLIP index not available, answering without context")
                else:
                    logger.info(f"Retrieving context (CLIP, k={k})...")
                    _, _, reports = self.clip_retriever.retrieve_by_image(image, k=k)
                    retrieved_reports = reports
                    context = " ".join(reports) if reports else None

            elif retriever == "colpali":
                if self.colpali_retriever is None:
                    logger.warning("ColPali index not available, answering without context")
                else:
                    logger.info(f"Retrieving context (ColPali, k={k})...")
                    _, _, reports = self.colpali_retriever.retrieve_by_image(image, k=k)
                    retrieved_reports = reports
                    context = " ".join(reports) if reports else None

        if self.medgemma is None:
            raise RuntimeError("MedGemma not available")

        logger.info(f"Generating answer...")
        answer = self.medgemma.answer_question(
            image,
            question,
            context=context,
            max_tokens=max_tokens
        )

        return {
            "question": question,
            "answer": answer,
            "context": context or "No context retrieved",
            "retrieved_reports": retrieved_reports,
            "retriever": retriever,
            "use_context": use_context,
        }

    def ablation_study(
        self,
        image: Image.Image,
        question: str,
        retriever: str = "clip",
        k: int = 3,
        max_tokens: int = 256
    ) -> Dict[str, Dict]:
        """
        Ablation: answer with and without context.

        Shows the value of RAG.

        Args:
            image: Input X-ray
            question: Clinical question
            retriever: "clip" or "colpali"
            k: Number of context docs
            max_tokens: Max answer length

        Returns:
            Dict with "with_context" and "without_context" keys
        """
        logger.info("Running ablation study...")
        logger.info("  1. Answering WITH context...")
        with_context = self.answer_question(
            image, question, retriever=retriever, use_context=True, k=k, max_tokens=max_tokens
        )

        logger.info("  2. Answering WITHOUT context...")
        without_context = self.answer_question(
            image, question, retriever=retriever, use_context=False, max_tokens=max_tokens
        )

        return {
            "with_context": with_context,
            "without_context": without_context,
        }

def main():
    import argparse

    parser = argparse.ArgumentParser(description="QA RAG Pipeline")
    parser.add_argument("--image", required=True, help="Path to X-ray image")
    parser.add_argument("--question", required=True, help="Clinical question")
    parser.add_argument("--retriever", choices=["clip", "colpali"], default="clip")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--clip_index", default=str(config.RESULTS_DIR / "clip_index"))
    parser.add_argument("--colpali_index", default=str(config.RESULTS_DIR / "colpali_index"))
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("QA RAG PIPELINE")
    logger.info("=" * 60)

    pipeline = QARagPipeline(
        device=args.device,
        use_4bit=config.MEDGEMMA_USE_4BIT,
        clip_index_path=args.clip_index,
        colpali_index_path=args.colpali_index if args.retriever == "colpali" else None,
        medgemma_backend=config.MEDGEMMA_BACKEND,
    )

    image = Image.open(args.image).convert("RGB")

    if args.ablation:
        result = pipeline.ablation_study(image, args.question, retriever=args.retriever)
        logger.info("\nWITH CONTEXT:")
        logger.info(result["with_context"]["answer"])
        logger.info("\nWITHOUT CONTEXT:")
        logger.info(result["without_context"]["answer"])
    else:
        result = pipeline.answer_question(image, args.question, retriever=args.retriever)
        logger.info(f"\nQuestion: {result['question']}")
        logger.info(f"\nContext: {result['context'][:500]}")
        logger.info(f"\nAnswer: {result['answer']}")

if __name__ == "__main__":
    main()
