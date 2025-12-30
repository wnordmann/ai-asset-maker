"""CLI entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from huggingface_hub import login

from ai_asset_maker.generate import GenerationConfig, generate_images
from ai_asset_maker.prompting import (
    PromptSpec,
    PromptVariant,
    build_prompt_variants,
    build_prompts,
    parse_fills,
)

# Set the path to your larger drive (e.g., your M: drive)
os.environ['HF_HOME'] = r"M:\huggingface_cache"

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
        required=False,
        help="Model ID or local path, e.g. runwayml/stable-diffusion-v1-5 or /models/sd15",
    )
    parser.add_argument(
        "--prompt",
        required=False,
        help="Prompt template with '{}' placeholders.",
    )
    parser.add_argument(
        "--fill",
        action="append",
        default=[],
        help="Comma-separated words for each '{}' placeholder. Use once per placeholder.",
    )
    parser.add_argument(
        "--json",
        help="JSON file path or inline JSON payload for named placeholders.",
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
    parser.add_argument("--height", type=int, default=512,
                        help="Image height in pixels.")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width in pixels.")
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


def _load_json_payload(raw: str) -> Dict[str, Any]:
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def _parse_json_spec(payload: Dict[str, Any]) -> tuple[PromptSpec, int, str]:
    if "model" not in payload or not isinstance(payload["model"], str):
        raise ValueError("JSON must include a string 'model' field.")
    if "prompt" not in payload or not isinstance(payload["prompt"], str):
        raise ValueError("JSON must include a string 'prompt' field.")
    if "seed" in payload or "seeds" in payload:
        raise ValueError("JSON seeds are not supported; use 'iterations' for random seeds.")
    iterations = payload.get("iterations", 1)
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("JSON 'iterations' must be an integer >= 1.")

    reserved = {"model", "prompt", "iterations"}
    fills_by_name: Dict[str, List[str]] = {}
    for key, value in payload.items():
        if key in reserved:
            continue
        if not isinstance(value, list) or not value:
            raise ValueError(f"JSON field '{key}' must be a non-empty list.")
        fills_by_name[key] = [str(item) for item in value]

    spec = PromptSpec(template=payload["prompt"], fills=[], fills_by_name=fills_by_name)
    return spec, iterations, payload["model"]


def _slugify(value: str) -> str:
    value = value.strip().lower().replace(" ", "-")
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in "-_")
    return cleaned or "value"


def _variant_output_dir(base_dir: Path, variant: PromptVariant) -> Path:
    if not variant.ordered_values:
        return base_dir / "default"
    parts = [f"{name}-{_slugify(val)}" for name, val in variant.ordered_values]
    dirname = "__".join(parts)
    if len(dirname) > 80:
        digest = hashlib.sha1(dirname.encode("utf-8")).hexdigest()[:10]
        dirname = f"variant_{digest}"
    return base_dir / dirname


def main() -> int:
    load_dotenv()
    token = (
        os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HF_ACCESS_TOKEN")
    )
    if token:
        login(token=token)
    parser = _build_parser()
    args = parser.parse_args()

    if args.json:
        if args.model:
            raise ValueError("Provide the model in JSON when using --json.")
        if args.prompt or args.fill:
            raise ValueError(
                "Provide prompt/fills in JSON or via CLI flags, not both.")
        if args.seed is not None or args.seeds is not None:
            raise ValueError(
                "Provide seeds in JSON or via CLI flags, not both.")
        payload = _load_json_payload(args.json)
        prompt_spec, iterations, model_path = _parse_json_spec(payload)
        variants = build_prompt_variants(prompt_spec)
        total_outputs = 0
        for variant in variants:
            seeds = [torch.seed() for _ in range(iterations)]
            output_dir = _variant_output_dir(Path(args.out), variant)
            config = GenerationConfig(
                model_path=model_path,
                output_dir=output_dir,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                device=args.device,
                fp16=args.fp16,
            )
            outputs = generate_images(config, [variant.prompt], seeds)
            total_outputs += len(outputs)
        print(f"Generated {total_outputs} image(s) in {Path(args.out)}.")
        return 0
    else:
        if not args.model:
            raise ValueError("--model is required unless --json is used.")
        if not args.prompt:
            raise ValueError("--prompt is required unless --json is used.")
        fills = parse_fills(args.fill)
        prompt_spec = PromptSpec(template=args.prompt, fills=fills)
        seeds = _parse_seeds(args.seed, args.seeds)

    prompts = build_prompts(prompt_spec)

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
