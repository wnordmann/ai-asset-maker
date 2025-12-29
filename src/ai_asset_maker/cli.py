"""CLI entrypoint."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from ai_asset_maker.generate import GenerationConfig, generate_images
from ai_asset_maker.prompting import PromptSpec, build_prompts, parse_fills


def _parse_seeds(seed: int | None, seeds: str | None) -> List[int]:
    if seed is not None and seeds is not None:
        raise ValueError("Use either --seed or --seeds, not both.")
    if seed is not None:
        return [seed]
    if seeds is not None:
        values = []
        for raw in seeds.split(","):
            raw = raw.strip()
            if not raw:
                continue
            values.append(int(raw))
        if not values:
            raise ValueError("--seeds must include at least one integer.")
        return values
    return [0]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Stable Diffusion images from prompt templates."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID or local path, e.g. runwayml/stable-diffusion-v1-5 or /models/sd15",
    )
    parser.add_argument("--prompt", required=True, help="Prompt template with '{}' placeholders.")
    parser.add_argument(
        "--fill",
        action="append",
        default=[],
        help="Comma-separated words for each '{}' placeholder. Use once per placeholder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Single seed for deterministic output (default: 0).",
    )
    parser.add_argument(
        "--seeds",
        help="Comma-separated list of seeds to iterate over.",
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory for generated images.",
    )
    parser.add_argument("--height", type=int, default=512, help="Image height in pixels.")
    parser.add_argument("--width", type=int, default=512, help="Image width in pixels.")
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 weights (recommended for CUDA).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    fills = parse_fills(args.fill)
    prompt_spec = PromptSpec(template=args.prompt, fills=fills)
    prompts = build_prompts(prompt_spec)
    seeds = _parse_seeds(args.seed, args.seeds)

    config = GenerationConfig(
        model_path=args.model,
        output_dir=Path(args.out),
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        device=args.device,
        fp16=args.fp16,
    )

    outputs = generate_images(config, prompts, seeds)
    print(f"Generated {len(outputs)} image(s) in {config.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
