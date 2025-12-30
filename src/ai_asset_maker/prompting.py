"""Prompt templating helpers."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import string
from typing import Dict, Iterable, List, Tuple

import os

# Set the path to your larger drive (e.g., your M: drive)
os.environ['HF_HOME'] = r"M:\huggingface_cache"


@dataclass(frozen=True)
class PromptSpec:
    template: str
    fills: List[List[str]]
    fills_by_name: Dict[str, List[str]] | None = None


@dataclass(frozen=True)
class PromptVariant:
    prompt: str
    ordered_values: List[Tuple[str, str]]


def _parse_placeholders(template: str) -> tuple[List[str], int]:
    formatter = string.Formatter()
    named_fields: List[str] = []
    unnamed_count = 0
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is not None:
            if field_name == "":
                unnamed_count += 1
            else:
                if field_name not in named_fields:
                    named_fields.append(field_name)
    if named_fields and unnamed_count:
        raise ValueError(
            "Do not mix named placeholders with '{}' placeholders.")
    return named_fields, unnamed_count


def parse_fills(fill_args: Iterable[str]) -> List[List[str]]:
    fills: List[List[str]] = []
    for raw in fill_args:
        words = [word.strip() for word in raw.split(",") if word.strip()]
        if not words:
            raise ValueError("Each --fill must include at least one word.")
        fills.append(words)
    return fills


def build_prompt_variants(spec: PromptSpec) -> List[PromptVariant]:
    named_fields, unnamed_count = _parse_placeholders(spec.template)
    if named_fields:
        if spec.fills:
            raise ValueError(
                "Named placeholders require JSON fills, not --fill.")
        fills_by_name = spec.fills_by_name or {}
        missing = [name for name in named_fields if name not in fills_by_name]
        if missing:
            raise ValueError(
                f"Missing fills for placeholder(s): {', '.join(missing)}.")
        variants: List[PromptVariant] = []
        value_lists = [fills_by_name[name] for name in named_fields]
        for combo in itertools.product(*value_lists):
            values = dict(zip(named_fields, combo))
            ordered_values = [(name, values[name]) for name in named_fields]
            variants.append(
                PromptVariant(
                    prompt=spec.template.format(**values),
                    ordered_values=ordered_values,
                )
            )
        return variants

    if unnamed_count == 0:
        if spec.fills:
            raise ValueError(
                "Template has no '{}' placeholders, but --fill was provided.")
        return [PromptVariant(prompt=spec.template, ordered_values=[])]

    if unnamed_count != len(spec.fills):
        raise ValueError(
            f"Template has {unnamed_count} placeholder(s), but {len(spec.fills)} --fill argument(s) were provided."
        )

    variants: List[PromptVariant] = []
    for combo in itertools.product(*spec.fills):
        variants.append(PromptVariant(
            prompt=spec.template.format(*combo), ordered_values=[]))
    return variants


def build_prompts(spec: PromptSpec) -> List[str]:
    return [variant.prompt for variant in build_prompt_variants(spec)]
