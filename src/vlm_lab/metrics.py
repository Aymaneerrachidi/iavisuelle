from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any

from .domain import MetricSummary, Precision


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def summarize(predictions: list[dict[str, Any]]) -> list[MetricSummary]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[(prediction["model_id"], prediction["precision"])].append(prediction)

    output: list[MetricSummary] = []
    for (model_id, precision), rows in grouped.items():
        hypotheses = [row["caption"] for row in rows]
        reference_sets = [row["references"] for row in rows]
        bleu4: float | None = None
        rouge_l: float | None = None
        if rows and all(refs for refs in reference_sets):
            try:
                import sacrebleu
                from rouge_score import rouge_scorer

                max_refs = max(len(refs) for refs in reference_sets)
                transposed = [
                    [refs[min(i, len(refs) - 1)] for refs in reference_sets]
                    for i in range(max_refs)
                ]
                bleu4 = sacrebleu.corpus_bleu(hypotheses, transposed).score
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                rouge_l = 100 * mean(
                    max(
                        scorer.score(reference, hypothesis)["rougeL"].fmeasure for reference in refs
                    )
                    for hypothesis, refs in zip(hypotheses, reference_sets, strict=True)
                )
            except ImportError:
                pass
        latencies = [float(row["latency_ms"]) for row in rows]
        vram = [float(row["peak_vram_mb"]) for row in rows if row["peak_vram_mb"] is not None]
        output.append(
            MetricSummary(
                model_id=model_id,
                precision=Precision(precision),
                samples=len(rows),
                bleu4=round(bleu4, 3) if bleu4 is not None else None,
                rouge_l=round(rouge_l, 3) if rouge_l is not None else None,
                mean_latency_ms=round(mean(latencies), 2),
                p95_latency_ms=round(percentile(latencies, 0.95), 2),
                peak_vram_mb=round(max(vram), 2) if vram else None,
                mean_caption_words=round(mean(len(text.split()) for text in hypotheses), 2),
            )
        )
    return output
