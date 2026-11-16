from __future__ import annotations

import gc
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any

from PIL import Image

from .domain import CaptionResult, GenerationOptions, Precision
from .model_registry import get_model


@dataclass
class LoadedModel:
    processor: Any
    model: Any
    device: str


class CaptionEngine:
    """Lazy, bounded model cache suitable for a single-GPU worker."""

    def __init__(self, cache_size: int = 1) -> None:
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, Precision], LoadedModel] = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def _imports() -> tuple[Any, Any, Any, Any, Any]:
        try:
            import psutil
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                BlipForConditionalGeneration,
                GitForCausalLM,
            )
        except ImportError as exc:
            raise RuntimeError(
                "ML dependencies are missing. Install with: pip install -e '.[ml]'"
            ) from exc
        return (
            torch,
            psutil,
            AutoProcessor,
            BitsAndBytesConfig,
            {
                "blip": BlipForConditionalGeneration,
                "git": GitForCausalLM,
            },
        )

    def _load(self, model_id: str, precision: Precision) -> LoadedModel:
        key = (model_id, precision)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        torch, _, auto_processor, quant_config_cls, model_classes = self._imports()
        info = get_model(model_id)
        if precision != Precision.FULL and not torch.cuda.is_available():
            raise RuntimeError("8-bit and 4-bit modes require a CUDA GPU and bitsandbytes.")

        load_args: dict[str, Any] = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if precision == Precision.INT8:
            load_args.update(
                quantization_config=quant_config_cls(load_in_8bit=True), device_map="auto"
            )
        elif precision == Precision.INT4:
            load_args.update(
                quantization_config=quant_config_cls(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                device_map="auto",
            )
        else:
            load_args["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

        processor = auto_processor.from_pretrained(info.hub_id)
        model = model_classes[info.architecture].from_pretrained(info.hub_id, **load_args)
        model.eval()
        if precision == Precision.FULL:
            model.to(device)

        loaded = LoadedModel(processor=processor, model=model, device=device)
        if self.cache_size > 0:
            self._cache[key] = loaded
            while len(self._cache) > self.cache_size:
                _, evicted = self._cache.popitem(last=False)
                del evicted
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return loaded

    def caption(
        self,
        image: Image.Image,
        model_id: str,
        precision: Precision,
        options: GenerationOptions,
    ) -> CaptionResult:
        with self._lock:
            torch, psutil, _, _, _ = self._imports()
            loaded = self._load(model_id, precision)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            inputs = loaded.processor(images=image.convert("RGB"), return_tensors="pt")
            if precision == Precision.FULL:
                moved = {}
                for key, value in inputs.items():
                    value = value.to(loaded.device)
                    if loaded.device == "cuda" and value.is_floating_point():
                        value = value.to(dtype=torch.float16)
                    moved[key] = value
                inputs = moved
            else:
                target_device = next(loaded.model.parameters()).device
                inputs = {key: value.to(target_device) for key, value in inputs.items()}

            started = time.perf_counter()
            with torch.inference_mode():
                output = loaded.model.generate(
                    **inputs,
                    max_new_tokens=options.max_new_tokens,
                    num_beams=options.num_beams,
                    temperature=options.temperature,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000
            caption = loaded.processor.batch_decode(output, skip_special_tokens=True)[0].strip()
            peak_vram = (
                torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
            )
            memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            return CaptionResult(
                caption=caption,
                model_id=model_id,
                precision=precision,
                latency_ms=round(latency_ms, 2),
                peak_vram_mb=round(peak_vram, 2) if peak_vram is not None else None,
                process_memory_mb=round(memory, 2),
            )
