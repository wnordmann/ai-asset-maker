# ai-asset-maker
Template-driven Stable Diffusion asset generator.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Single prompt, single seed:

```bash
python -m ai_asset_maker.cli \
  --model /path/to/model \
  --prompt "A cinematic portrait of a knight" \
  --seed 42
```

Prompt with wildcard slots and multiple seeds:

```bash
python -m ai_asset_maker.cli \
  --model /path/to/model \
  --prompt "A {} in a {} at golden hour" \
  --fill "cat,dog,fox" \
  --fill "forest,city,desert" \
  --seeds "11,22,33"
```

Notes:
- Use `{}` placeholders in `--prompt`. Provide one `--fill` per placeholder.
- Output images are written to `outputs/` by default.
