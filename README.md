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
python -m ai_asset_maker.cli --model M:\models_backup\diffusion_models\v1-5-pruned-emaonly.safetensors --prompt "A {} in a {} at golden hour" --fill "cat,dog,fox" --fill "forest,city,desert" --seeds "11,22,33"
```
```M:\models_backup\diffusion_models\v1-5-pruned-emaonly.safetensors```
Notes:
- Use `{}` placeholders in `--prompt`. Provide one `--fill` per placeholder.
- Output images are written to `outputs/` by default.

## CUDA notes (Windows)

If `torch.cuda.is_available()` is `False`, you likely have a CPU-only PyTorch build.
Your NVIDIA driver can be newer than PyTorch's CUDA wheels (driver is backward-compatible).

Check your driver CUDA version:

```bash
nvidia-smi
```

Install a CUDA-enabled PyTorch build (inside your venv):

```bash
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

or:

```bash
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA from Python:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Note: `nvcc --version` only works if the CUDA Toolkit is installed; it's not required to run PyTorch on the GPU.
