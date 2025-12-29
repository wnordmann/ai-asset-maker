"""Stable Diffusion image generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import hashlib

import torch
from diffusers import StableDiffusionPipeline


@dataclass(frozen=True)
class GenerationConfig:
    model_path: str
    output_dir: Path
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    device: str
    fp16: bool


def _prompt_slug(prompt: str) -> str:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return digest


def _build_pipeline(model_path: str, device: str, fp16: bool) -> StableDiffusionPipeline:
    dtype = torch.float16 if fp16 else torch.float32
    path = Path(model_path)
    if path.is_file():
        pipe = StableDiffusionPipeline.from_single_file(str(path), torch_dtype=dtype)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def generate_images(
    config: GenerationConfig,
    prompts: Iterable[str],
    seeds: Iterable[int],
) -> List[Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = _build_pipeline(config.model_path, config.device, config.fp16)

    outputs: List[Path] = []
    for prompt_index, prompt in enumerate(prompts, start=1):
        prompt_hash = _prompt_slug(prompt)
        for seed in seeds:
            generator = torch.Generator(device=config.device).manual_seed(seed)
            result = pipe(
                prompt,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            )
            image = result.images[0]
            filename = f"p{prompt_index:03d}_{prompt_hash}_seed{seed}.png"
            output_path = config.output_dir / filename
            image.save(output_path)
            outputs.append(output_path)
    return outputs
