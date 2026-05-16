"""
ColPali-based multi-vector retrieval index.

ColPali (PaliGemma-3B) generates ColBERT-style multi-vector embeddings for
fine-grained late-interaction matching between text queries and document images.

NOTE: ColPali needs a GPU with ~6-8 GB VRAM — it will NOT fit on a 4 GB GTX 1650.
Run this on Colab / Kaggle (free T4) and commit the produced index back to the repo.

Usage:
    python src/models/colpali_index.py --mode build --corpus_path data/processed/corpus.csv
    python src/models/colpali_index.py --mode retrieve --query_image path/to/image.jpg
"""

import logging
import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ColPaliRetriever:
    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize ColPali model and processor.

        Args:
            model_name: HuggingFace model ID for ColPali
            device: Device to load model on (ColPali needs "cuda")
            torch_dtype: Data type (default: bfloat16 on GPU, float32 on CPU)
        """
        self.model_name = model_name
        self.device = device

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        logger.info(f"Loading ColPali ({model_name}) on {device} ...")
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor
        except ImportError as e:
            logger.error("colpali-engine not installed. Run: pip install colpali-engine")
            raise e

        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device if device == "cuda" else None,
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)

        self.embeddings: List[torch.Tensor] = []
        self.metadata: Optional[dict] = None

        logger.info("ColPali model loaded")

    def encode_images(self, images: List[Image.Image], batch_size: int = 4) -> List[torch.Tensor]:
        """Encode a list of images into multi-vector embeddings (one tensor per image)."""
        all_embeddings: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_inputs = self.processor.process_images(batch).to(self.model.device)
            with torch.no_grad():
                batch_emb = self.model(**batch_inputs)
            for emb in batch_emb:
                all_embeddings.append(emb.cpu().to(torch.float32))
        return all_embeddings

    def encode_queries(self, queries: List[str], batch_size: int = 8) -> List[torch.Tensor]:
        """Encode a list of text queries into multi-vector embeddings."""
        all_embeddings: List[torch.Tensor] = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            batch_inputs = self.processor.process_queries(batch).to(self.model.device)
            with torch.no_grad():
                batch_emb = self.model(**batch_inputs)
            for emb in batch_emb:
                all_embeddings.append(emb.cpu().to(torch.float32))
        return all_embeddings

    def build_index(self, corpus_df: pd.DataFrame, index_path: Optional[str] = None) -> None:
        """
        Build the multi-vector index from a corpus.

        Args:
            corpus_df: DataFrame with columns study_id, image_path, text
            index_path: Directory to save embeddings + metadata
        """
        if index_path is None:
            index_path = config.RESULTS_DIR / "colpali_index"
        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Building ColPali index from {len(corpus_df)} studies ...")

        images: List[Image.Image] = []
        valid_indices: List[int] = []
        for idx, row in corpus_df.iterrows():
            image_path = row.get("image_path", None)
            if image_path is None or pd.isna(image_path) or not Path(str(image_path)).exists():
                logger.warning(f"Image not found for row {idx}: {image_path}, skipping")
                continue
            try:
                images.append(Image.open(image_path).convert("RGB"))
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")

        logger.info(f"Encoding {len(images)} images with ColPali ...")
        self.embeddings = self.encode_images(images)

        self.metadata = {
            "study_ids": corpus_df.iloc[valid_indices]["study_id"].tolist(),
            "image_paths": corpus_df.iloc[valid_indices]["image_path"].tolist(),
            "reports": corpus_df.iloc[valid_indices]["text"].tolist(),
            "valid_indices": valid_indices,
        }

        torch.save(self.embeddings, index_path / "colpali_embeddings.pt")
        with open(index_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved ColPali index ({len(self.embeddings)} docs) to {index_path}")

    def load_index(self, index_path: str) -> None:
        """Load a prebuilt ColPali index."""
        index_path = Path(index_path)
        self.embeddings = torch.load(index_path / "colpali_embeddings.pt")
        with open(index_path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"Loaded ColPali index ({len(self.embeddings)} docs) from {index_path}")

    def _score(self, query_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Score query embeddings against the indexed document embeddings (MaxSim)."""
        if not self.embeddings:
            raise RuntimeError("Index not built/loaded")
        return self.processor.score_multi_vector(query_embeddings, self.embeddings)

    def retrieve_by_text(self, text: str, k: int = 5) -> Tuple[List[str], List[float], List[str]]:
        """Retrieve top-k studies for a text query."""
        query_emb = self.encode_queries([text])
        scores = self._score(query_emb)[0]
        topk = torch.topk(scores, min(k, len(scores)))
        indices = topk.indices.tolist()
        study_ids = [self.metadata["study_ids"][i] for i in indices]
        reports = [self.metadata["reports"][i] for i in indices]
        return study_ids, topk.values.tolist(), reports

    def retrieve_by_image(self, image: Image.Image, k: int = 5) -> Tuple[List[str], List[float], List[str]]:
        """Retrieve top-k similar studies for a query image (image-as-query MaxSim)."""
        query_emb = self.encode_images([image])
        scores = self._score(query_emb)[0]
        topk = torch.topk(scores, min(k, len(scores)))
        indices = topk.indices.tolist()
        study_ids = [self.metadata["study_ids"][i] for i in indices]
        reports = [self.metadata["reports"][i] for i in indices]
        return study_ids, topk.values.tolist(), reports


def main():
    parser = argparse.ArgumentParser(description="ColPali multi-vector retrieval index")
    parser.add_argument("--mode", choices=["build", "retrieve"], required=True)
    parser.add_argument("--corpus_path", default=str(config.PROCESSED_DATA_DIR / "corpus.csv"))
    parser.add_argument("--index_path", default=str(config.RESULTS_DIR / "colpali_index"))
    parser.add_argument("--query_image", help="Image path for retrieval")
    parser.add_argument("--query_text", help="Text query for retrieval")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("COLPALI RETRIEVAL INDEX")
    logger.info("=" * 60)

    if args.device != "cuda":
        logger.warning("ColPali strongly prefers a CUDA GPU. CPU will be very slow / may OOM.")

    retriever = ColPaliRetriever(device=args.device)

    if args.mode == "build":
        corpus_df = pd.read_csv(args.corpus_path)
        retriever.build_index(corpus_df, args.index_path)

    elif args.mode == "retrieve":
        retriever.load_index(args.index_path)
        if args.query_image:
            image = Image.open(args.query_image).convert("RGB")
            study_ids, scores, reports = retriever.retrieve_by_image(image, k=args.k)
        elif args.query_text:
            study_ids, scores, reports = retriever.retrieve_by_text(args.query_text, k=args.k)
        else:
            raise ValueError("Provide --query_image or --query_text for retrieve mode")

        logger.info("\nTop results:")
        for i, (sid, score, report) in enumerate(zip(study_ids, scores, reports)):
            logger.info(f"{i+1}. Study {sid} (score: {score:.4f})")
            logger.info(f"   {report[:200]}...")


if __name__ == "__main__":
    main()
