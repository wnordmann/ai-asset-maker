"""Prompt templating helpers."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import string
from typing import Iterable, List


@dataclass(frozen=True)
class PromptSpec:
    template: str
    fills: List[List[str]]


def _count_placeholders(template: str) -> int:
    formatter = string.Formatter()
    count = 0
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is not None:
            if field_name == "":
                count += 1
            else:
                raise ValueError(
                    "Named placeholders are not supported; use '{}' placeholders only."
                )
    return count


def parse_fills(fill_args: Iterable[str]) -> List[List[str]]:
    fills: List[List[str]] = []
    for raw in fill_args:
        words = [word.strip() for word in raw.split(",") if word.strip()]
        if not words:
            raise ValueError("Each --fill must include at least one word.")
        fills.append(words)
    return fills


def build_prompts(spec: PromptSpec) -> List[str]:
    placeholder_count = _count_placeholders(spec.template)
    if placeholder_count == 0:
        if spec.fills:
            raise ValueError("Template has no '{}' placeholders, but --fill was provided.")
        return [spec.template]

    if placeholder_count != len(spec.fills):
        raise ValueError(
            f"Template has {placeholder_count} placeholder(s), but {len(spec.fills)} --fill argument(s) were provided."
        )

    prompts: List[str] = []
    for combo in itertools.product(*spec.fills):
        prompts.append(spec.template.format(*combo))
    return prompts
