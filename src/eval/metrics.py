"""
Evaluation metrics for report generation and QA.

Metrics:
  - Report Gen: BLEU, ROUGE, METEOR, BERTScore
  - Retrieval: Recall@k, MRR
  - QA: Token F1, BERTScore
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import torch
from bert_score import score as bert_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenMetrics:
    """Metrics for report generation evaluation."""

    @staticmethod
    def bleu(reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
        """
        Calculate BLEU score.

        Args:
            reference: Ground truth report
            hypothesis: Generated report
            max_n: Max n-gram (default: 4 for BLEU-4)

        Returns:
            Dict with bleu_1, bleu_2, bleu_3, bleu_4
        """
        try:
            ref_tokens = word_tokenize(reference.lower())
            hyp_tokens = word_tokenize(hypothesis.lower())

            smoothing = SmoothingFunction().method1

            scores = {}
            for n in range(1, max_n + 1):
                score = sentence_bleu(
                    [ref_tokens],
                    hyp_tokens,
                    weights=[1.0 / n] * n,
                    smoothing_function=smoothing
                )
                scores[f"bleu_{n}"] = score

            return scores
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}

    @staticmethod
    def rouge(reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.

        Args:
            reference: Ground truth
            hypothesis: Generated text

        Returns:
            Dict with rouge1, rouge2, rougeL
        """
        try:
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)

            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    @staticmethod
    def bertscore(references: List[str], hypotheses: List[str], lang: str = "en") -> Dict[str, float]:
        """
        Calculate BERTScore.

        Args:
            references: List of reference texts
            hypotheses: List of generated texts
            lang: Language code

        Returns:
            Dict with precision, recall, f1
        """
        try:
            P, R, F1 = bert_score(hypotheses, references, lang=lang, verbose=False)
            return {
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
                "bertscore_f1": F1.mean().item(),
            }
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

class RetrievalMetrics:
    """Metrics for retrieval evaluation."""

    @staticmethod
    def recall_at_k(
        relevant_indices: List[int],
        retrieved_indices: List[int],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@k.

        Args:
            relevant_indices: Ground truth relevant document indices
            retrieved_indices: Retrieved document indices (ranked)
            k: Cutoff

        Returns:
            Recall@k score
        """
        if not relevant_indices:
            return 0.0

        retrieved_k = set(retrieved_indices[:k])
        relevant_k = set(relevant_indices)
        intersection = len(retrieved_k & relevant_k)
        recall = intersection / len(relevant_k)
        return recall

    @staticmethod
    def mrr(relevant_indices: List[int], retrieved_indices: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            relevant_indices: Ground truth relevant indices
            retrieved_indices: Retrieved indices (ranked)

        Returns:
            MRR score
        """
        relevant_set = set(relevant_indices)
        for rank, idx in enumerate(retrieved_indices, 1):
            if idx in relevant_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def batch_recall_at_k(
        all_relevant: List[List[int]],
        all_retrieved: List[List[int]],
        k: int = 5
    ) -> float:
        """Batch Recall@k."""
        scores = [
            RetrievalMetrics.recall_at_k(rel, ret, k)
            for rel, ret in zip(all_relevant, all_retrieved)
        ]
        return np.mean(scores) if scores else 0.0

class QAMetrics:
    """Metrics for QA evaluation."""

    @staticmethod
    def token_f1(reference: str, hypothesis: str) -> float:
        """
        Calculate token-level F1 score.

        Args:
            reference: Gold answer
            hypothesis: Generated answer

        Returns:
            F1 score
        """
        try:
            ref_tokens = set(word_tokenize(reference.lower()))
            hyp_tokens = set(word_tokenize(hypothesis.lower()))

            if not ref_tokens or not hyp_tokens:
                return 0.0

            intersection = len(ref_tokens & hyp_tokens)
            precision = intersection / len(hyp_tokens) if hyp_tokens else 0.0
            recall = intersection / len(ref_tokens) if ref_tokens else 0.0

            if precision + recall == 0:
                return 0.0

            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
        except Exception as e:
            logger.warning(f"Token F1 calculation failed: {e}")
            return 0.0

    @staticmethod
    def exact_match(reference: str, hypothesis: str) -> int:
        """Exact match (0 or 1)."""
        return 1 if reference.lower().strip() == hypothesis.lower().strip() else 0
