"""
Run evaluation across report generation, retrieval, and QA modes.

Reads the artifacts produced by the Colab notebook + the local CLIP index, then
writes comparison tables (markdown + CSV) to `results/comparison_tables/`.

Inputs expected (any subset is fine; the script reports on what's available):
  results/generated_reports.csv   columns: study_id, ground_truth, medgemma_direct
                                  (optionally clip_retrieval, colpali_retrieval)
  results/qa_answers.csv          columns: study_id, question, gold_answer,
                                  answer_no_context, answer_with_context

Usage:
    python src/eval/run_eval.py
    python src/eval/run_eval.py --reports results/generated_reports.csv \\
                                --qa results/qa_answers.csv \\
                                --out results/comparison_tables
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from src.eval.metrics import ReportGenMetrics, QAMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _safe_text(x) -> str:
    return "" if x is None or pd.isna(x) else str(x)


def evaluate_report_column(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    """Compute BLEU-1..4, ROUGE-1/2/L, BERTScore averaged across the eval set."""
    pairs = [(r, h) for r, h in zip(refs, hyps) if r and h]
    if not pairs:
        return {}
    refs_ok = [r for r, _ in pairs]
    hyps_ok = [h for _, h in pairs]

    bleu_agg = {f"bleu_{n}": 0.0 for n in range(1, 5)}
    rouge_agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for r, h in pairs:
        for k, v in ReportGenMetrics.bleu(r, h).items():
            bleu_agg[k] += v
        for k, v in ReportGenMetrics.rouge(r, h).items():
            rouge_agg[k] += v
    n = len(pairs)
    for k in bleu_agg:
        bleu_agg[k] /= n
    for k in rouge_agg:
        rouge_agg[k] /= n

    bert = ReportGenMetrics.bertscore(refs_ok, hyps_ok)

    return {**bleu_agg, **rouge_agg, **bert, "n_samples": n}


def eval_reports(reports_csv: Path, out_dir: Path):
    """Evaluate every prediction column found in the reports CSV."""
    if not reports_csv.exists():
        logger.warning(f"Report file not found: {reports_csv}, skipping")
        return None

    df = pd.read_csv(reports_csv)
    logger.info(f"Loaded {len(df)} report rows from {reports_csv}")

    if "ground_truth" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "ground_truth"})
    if "ground_truth" not in df.columns:
        logger.warning("No ground_truth column found; cannot evaluate reports.")
        return None

    refs = df["ground_truth"].fillna("").astype(str).tolist()
    candidate_cols = [c for c in df.columns
                      if c not in ("ground_truth", "study_id", "image_path", "text")]
    candidate_cols = [c for c in candidate_cols
                      if df[c].apply(lambda x: isinstance(x, str)).any()]

    if not candidate_cols:
        logger.warning("No prediction columns found in reports CSV.")
        return None

    rows = []
    for col in candidate_cols:
        hyps = df[col].fillna("").astype(str).tolist()
        logger.info(f"Evaluating prediction column: {col}")
        metrics = evaluate_report_column(refs, hyps)
        if not metrics:
            continue
        metrics["approach"] = col
        rows.append(metrics)

    if not rows:
        return None

    result = pd.DataFrame(rows).set_index("approach")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "report_generation_comparison.csv")
    (out_dir / "report_generation_comparison.md").write_text(
        "# Report Generation - Model Comparison\n\n"
        + result.round(4).to_markdown() + "\n",
        encoding="utf-8",
    )
    logger.info(f"Saved report comparison to {out_dir}/report_generation_comparison.*")
    return result


def eval_qa(qa_csv: Path, out_dir: Path):
    """Evaluate QA: token F1, exact match, BERTScore for each answer column."""
    if not qa_csv.exists():
        logger.warning(f"QA answers file not found: {qa_csv}, skipping")
        return None

    df = pd.read_csv(qa_csv)
    logger.info(f"Loaded {len(df)} QA rows from {qa_csv}")

    if "gold_answer" not in df.columns and "answer" in df.columns:
        df = df.rename(columns={"answer": "gold_answer"})
    if "gold_answer" not in df.columns:
        logger.warning("No gold_answer column; cannot evaluate QA.")
        return None

    refs = df["gold_answer"].fillna("").astype(str).tolist()
    pred_cols = [c for c in df.columns if c.startswith("answer_")]
    if not pred_cols:
        logger.warning("No prediction columns (answer_*) found in QA CSV.")
        return None

    rows = []
    for col in pred_cols:
        hyps = df[col].fillna("").astype(str).tolist()
        f1s = [QAMetrics.token_f1(r, h) for r, h in zip(refs, hyps) if r and h]
        ems = [QAMetrics.exact_match(r, h) for r, h in zip(refs, hyps) if r and h]
        pairs = [(r, h) for r, h in zip(refs, hyps) if r and h]
        if not pairs:
            continue
        bert = ReportGenMetrics.bertscore([r for r, _ in pairs], [h for _, h in pairs])
        rows.append({
            "condition": col,
            "token_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "exact_match": sum(ems) / len(ems) if ems else 0.0,
            "n_samples": len(pairs),
            **bert,
        })

    if not rows:
        return None

    result = pd.DataFrame(rows).set_index("condition")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "qa_comparison.csv")
    (out_dir / "qa_comparison.md").write_text(
        "# QA - With vs Without RAG Context\n\n"
        + result.round(4).to_markdown() + "\n",
        encoding="utf-8",
    )
    logger.info(f"Saved QA comparison to {out_dir}/qa_comparison.*")
    return result


def write_summary(report_df: Optional[pd.DataFrame], qa_df: Optional[pd.DataFrame],
                  out_dir: Path):
    """Write a top-level summary markdown that bundles both tables."""
    lines = ["# Evaluation Summary\n"]
    if report_df is not None:
        lines += ["## Report Generation\n", report_df.round(4).to_markdown(), "\n"]
    if qa_df is not None:
        lines += ["## QA (RAG ablation)\n", qa_df.round(4).to_markdown(), "\n"]
    if report_df is None and qa_df is None:
        lines += ["_No artifacts found to evaluate._\n"]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {out_dir / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluation orchestrator")
    parser.add_argument("--reports", default=str(config.RESULTS_DIR / "generated_reports.csv"))
    parser.add_argument("--qa", default=str(config.RESULTS_DIR / "qa_answers.csv"))
    parser.add_argument("--out", default=str(config.COMPARISON_DIR))
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EVALUATION RUN")
    logger.info("=" * 60)
    out_dir = Path(args.out)

    report_df = eval_reports(Path(args.reports), out_dir)
    qa_df = eval_qa(Path(args.qa), out_dir)
    write_summary(report_df, qa_df, out_dir)

    logger.info("=" * 60)
    logger.info(f"DONE. Tables in {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
