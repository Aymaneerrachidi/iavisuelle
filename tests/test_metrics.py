from vlm_lab.metrics import summarize


def test_summary_computes_runtime_metrics_without_optional_quality_packages():
    rows = [
        {
            "model_id": "blip-base",
            "precision": "full",
            "caption": "a dog runs",
            "references": ["a dog runs"],
            "latency_ms": 10,
            "peak_vram_mb": 200,
        },
        {
            "model_id": "blip-base",
            "precision": "full",
            "caption": "a cat sits",
            "references": ["a cat sits"],
            "latency_ms": 30,
            "peak_vram_mb": 220,
        },
    ]
    result = summarize(rows)[0]
    assert result.samples == 2
    assert result.mean_latency_ms == 20
    assert result.p95_latency_ms == 30
    assert result.peak_vram_mb == 220
    assert result.mean_caption_words == 3
