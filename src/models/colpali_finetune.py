"""
ColPali LoRA fine-tuning on chest X-ray (image, report) pairs.  [STRETCH GOAL]

Fine-tunes ColPali with LoRA adapters using an in-batch contrastive loss so that
a report query embedding is pulled towards its matching X-ray image embedding.

NOTE: Requires a GPU (~12-16 GB recommended). Run on Colab / Kaggle, not the GTX 1650.

Usage:
    python src/models/colpali_finetune.py --corpus_path data/processed/corpus.csv \
        --num_epochs 3 --lora_r 32 --batch_size 4
"""

import logging
import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CXRPairDataset(Dataset):
    """(image, report) pairs from the corpus."""

    def __init__(self, corpus_df: pd.DataFrame):
        self.rows = []
        for _, row in corpus_df.iterrows():
            image_path = row.get("image_path", None)
            report = row.get("text", "")
            if image_path is None or pd.isna(image_path) or not Path(str(image_path)).exists():
                continue
            if not report or len(str(report).strip()) < 10:
                continue
            self.rows.append({"image_path": str(image_path), "report": str(report)})
        logger.info(f"Dataset built with {len(self.rows)} valid (image, report) pairs")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return Image.open(row["image_path"]).convert("RGB"), row["report"]


def contrastive_loss(query_emb: torch.Tensor, doc_scores: torch.Tensor) -> torch.Tensor:
    """
    In-batch contrastive loss. doc_scores is a [B, B] MaxSim score matrix where the
    diagonal holds the positive (matching) pairs. Cross-entropy pulls the diagonal up.
    """
    targets = torch.arange(doc_scores.size(0), device=doc_scores.device)
    return torch.nn.functional.cross_entropy(doc_scores, targets)


def finetune_colpali(
    corpus_path: str,
    output_dir: Optional[str] = None,
    num_epochs: int = 3,
    lora_r: int = 32,
    lora_alpha: int = 32,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    device: str = "cuda",
) -> str:
    """
    LoRA fine-tune ColPali on CXR (image, report) pairs.

    Returns:
        Path to the saved LoRA adapter directory.
    """
    from colpali_engine.models import ColPali, ColPaliProcessor
    from peft import LoraConfig, get_peft_model

    if output_dir is None:
        output_dir = config.RESULTS_DIR / "colpali_finetuned"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base ColPali model ...")
    model = ColPali.from_pretrained(
        config.MODEL_IDS["colpali"],
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = ColPaliProcessor.from_pretrained(config.MODEL_IDS["colpali"])

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    corpus_df = pd.read_csv(corpus_path)
    dataset = CXRPairDataset(corpus_df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: ([b[0] for b in batch], [b[1] for b in batch]),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )

    logger.info(f"Starting fine-tune: {num_epochs} epochs, {len(loader)} steps/epoch")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, (images, reports) in enumerate(loader):
            image_inputs = processor.process_images(images).to(model.device)
            query_inputs = processor.process_queries(reports).to(model.device)

            image_emb = model(**image_inputs)
            query_emb = model(**query_inputs)

            scores = processor.score_multi_vector(query_emb, image_emb)
            loss = contrastive_loss(query_emb, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                logger.info(f"  Epoch {epoch+1} | Step {step}/{len(loader)} | Loss {loss.item():.4f}")

        logger.info(f"Epoch {epoch+1} done | Avg loss {epoch_loss / max(len(loader), 1):.4f}")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Saved fine-tuned LoRA adapter to {output_dir}")
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune ColPali on CXR pairs")
    parser.add_argument("--corpus_path", default=str(config.PROCESSED_DATA_DIR / "corpus.csv"))
    parser.add_argument("--output_dir", default=str(config.RESULTS_DIR / "colpali_finetuned"))
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("COLPALI LORA FINE-TUNE (STRETCH GOAL)")
    logger.info("=" * 60)

    if args.device != "cuda":
        logger.warning("Fine-tuning ColPali on CPU is impractical. Use a GPU (Colab/Kaggle).")

    finetune_colpali(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
