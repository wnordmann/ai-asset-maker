"""Stable Diffusion image generation."""
from __future__ import annotations
from diffusers import AutoPipelineForText2Image
import torch

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import hashlib

import os

# Set the path to your larger drive (e.g., your M: drive)
os.environ['HF_HOME'] = r"M:\huggingface_cache"


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


def _build_pipeline(model_path: str, device: str, fp16: bool) -> AutoPipelineForText2Image:
    # dtype = torch.float16 if fp16 else torch.float32
    dtype = torch.float16
    path = Path(model_path)
    if path.is_file():
        try:
            if not hasattr(AutoPipelineForText2Image, "from_single_file"):
                raise ValueError(
                    "This diffusers version does not support from_single_file. "
                    "Upgrade diffusers or use a repo ID/local diffusers directory."
                )
            pipe = AutoPipelineForText2Image.from_single_file(
                str(path), torch_dtype=dtype)
        except Exception as exc:  # pragma: no cover - surface actionable error to user
            raise ValueError(
                "Single-file checkpoints are not supported for this model. "
                "Use a Hugging Face repo ID or a local diffusers directory."
            ) from exc
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_path, torch_dtype=dtype)
    # pipe.enable_model_cpu_offload()
    # pipe = pipe.to(device)
    # if device == "cuda":
    #     pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")
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
            metadata_path = output_path.with_suffix(".txt")
            metadata = "\n".join(
                [
                    f"prompt: {prompt}",
                    f"seed: {seed}",
                    f"model: {config.model_path}",
                    f"size: {config.width}x{config.height}",
                    f"steps: {config.num_inference_steps}",
                    f"guidance: {config.guidance_scale}",
                    f"device: {config.device}",
                    f"fp16: {config.fp16}",
                ]
            )
            metadata_path.write_text(metadata, encoding="utf-8")
            outputs.append(output_path)
    return outputs
