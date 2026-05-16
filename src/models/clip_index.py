"""
CLIP-based image-text retrieval index.

Light-weight option that runs on CPU/small GPUs.
Can be used for both report generation (retrieve similar X-rays) and QA (retrieve context).

Usage:
    python src/models/clip_index.py --mode build --corpus_path data/processed/corpus.csv
    python src/models/clip_index.py --mode retrieve --query_image path/to/image.jpg
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from PIL import Image
import open_clip
import faiss
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.data.download import get_dataset_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CLIPRetriever:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
        """
        Initialize CLIP model and FAISS index.

        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights to load
            device: Device to load model on ("cpu" or "cuda")
        """
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained

        logger.info(f"Loading CLIP {model_name} ({pretrained}) on {device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        self.embedding_dim = self.model.visual.output_dim
        self.index = None
        self.metadata = None

        logger.info(f"CLIP model loaded | Embedding dim: {self.embedding_dim}")

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode an image to embedding."""
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
        return embedding.cpu().numpy().astype(np.float32)[0]

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding."""
        tokens = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
        return embedding.cpu().numpy().astype(np.float32)[0]

    def build_index(self, corpus_df: pd.DataFrame, index_path: Optional[str] = None) -> None:
        """
        Build FAISS index from corpus.

        Args:
            corpus_df: DataFrame with columns: study_id, image_path, text
            index_path: Where to save index and metadata
        """
        logger.info(f"Building CLIP index from {len(corpus_df)} studies...")

        if index_path is None:
            index_path = config.RESULTS_DIR / "clip_index"
        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        embeddings = []
        valid_indices = []
        dataset_root = get_dataset_root()
        # collect candidate base directories (dataset_root itself + immediate subdirs)
        _search_bases = [dataset_root] + [d for d in dataset_root.iterdir() if d.is_dir()]

        for idx, row in corpus_df.iterrows():
            if idx % 50 == 0:
                logger.info(f"  Processing {idx}/{len(corpus_df)}...")

            try:
                if "image_path" in row and pd.notna(row["image_path"]):
                    image_path = Path(row["image_path"])
                    if not image_path.exists():
                        # For absolute paths, extract the relative tail starting at "files/"
                        if image_path.is_absolute():
                            parts = image_path.parts
                            try:
                                files_idx = next(i for i, p in enumerate(parts) if p == "files")
                                rel = Path(*parts[files_idx:])
                            except StopIteration:
                                rel = Path(image_path.name)
                        else:
                            rel = image_path
                        resolved = None
                        for base in _search_bases:
                            candidate = base / rel
                            if candidate.exists():
                                resolved = candidate
                                break
                        if resolved is None:
                            matches = list(dataset_root.rglob(image_path.name))
                            resolved = matches[0] if matches else image_path
                        image_path = resolved
                    if image_path.exists():
                        image = Image.open(image_path).convert("RGB")
                        embedding = self.encode_image(image)
                    else:
                        logger.warning(f"Image not found: {image_path}, skipping")
                        continue
                else:
                    logger.warning(f"No image_path for row {idx}, skipping")
                    continue

                embeddings.append(embedding)
                valid_indices.append(idx)

            except Exception as e:
                logger.warning(f"Failed to process row {idx}: {e}")
                continue

        embeddings = np.array(embeddings)
        logger.info(f"Built embeddings for {len(embeddings)} studies")

        if len(embeddings) == 0:
            raise RuntimeError("No valid images found — check that image paths in corpus.csv resolve correctly.")

        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)

        self.metadata = {
            "study_ids": corpus_df.iloc[valid_indices]["study_id"].tolist(),
            "image_paths": corpus_df.iloc[valid_indices]["image_path"].tolist(),
            "reports": corpus_df.iloc[valid_indices]["text"].tolist(),
            "valid_indices": valid_indices,
        }

        faiss.write_index(self.index, str(index_path / "clip.index"))
        with open(index_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved CLIP index to {index_path}")

    def load_index(self, index_path: str) -> None:
        """Load prebuilt index and metadata."""
        index_path = Path(index_path)
        self.index = faiss.read_index(str(index_path / "clip.index"))
        with open(index_path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"Loaded CLIP index from {index_path}")

    def retrieve_by_image(self, image: Image.Image, k: int = 5) -> Tuple[List[str], List[float], List[str]]:
        """
        Retrieve top-k similar reports given an image.

        Args:
            image: Query image
            k: Number of results to return

        Returns:
            (study_ids, distances, reports)
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index not built/loaded")

        embedding = self.encode_image(image).reshape(1, -1)
        distances, indices = self.index.search(embedding, k)

        study_ids = [self.metadata["study_ids"][i] for i in indices[0]]
        reports = [self.metadata["reports"][i] for i in indices[0]]
        distances = distances[0].tolist()

        return study_ids, distances, reports

    def retrieve_by_text(self, text: str, k: int = 5) -> Tuple[List[str], List[float], List[str]]:
        """
        Retrieve top-k studies given a text query.

        Args:
            text: Query text
            k: Number of results to return

        Returns:
            (study_ids, distances, reports)
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index not built/loaded")

        embedding = self.encode_text(text).reshape(1, -1)
        distances, indices = self.index.search(embedding, k)

        study_ids = [self.metadata["study_ids"][i] for i in indices[0]]
        reports = [self.metadata["reports"][i] for i in indices[0]]
        distances = distances[0].tolist()

        return study_ids, distances, reports

def main():
    parser = argparse.ArgumentParser(description="CLIP-based retrieval index")
    parser.add_argument("--mode", choices=["build", "retrieve"], required=True)
    parser.add_argument("--corpus_path", default=str(config.PROCESSED_DATA_DIR / "corpus.csv"))
    parser.add_argument("--index_path", default=str(config.RESULTS_DIR / "clip_index"))
    parser.add_argument("--query_image", help="Image path for retrieval")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CLIP RETRIEVAL INDEX")
    logger.info("=" * 60)

    retriever = CLIPRetriever(device=args.device)

    if args.mode == "build":
        logger.info(f"Building index from {args.corpus_path}...")
        corpus_df = pd.read_csv(args.corpus_path)
        retriever.build_index(corpus_df, args.index_path)

    elif args.mode == "retrieve":
        logger.info(f"Loading index from {args.index_path}...")
        retriever.load_index(args.index_path)

        if args.query_image:
            logger.info(f"Retrieving similar studies for {args.query_image}...")
            image = Image.open(args.query_image).convert("RGB")
            study_ids, distances, reports = retriever.retrieve_by_image(image, k=args.k)

            logger.info("\nTop results:")
            for i, (sid, dist, report) in enumerate(zip(study_ids, distances, reports)):
                logger.info(f"{i+1}. Study {sid} (distance: {dist:.4f})")
                logger.info(f"   {report[:200]}...")

if __name__ == "__main__":
    main()
