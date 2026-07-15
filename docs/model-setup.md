# Model Setup for the 5090 Path

Week 1 has three non-optional model components for the real demo:

- PaddleOCR / PP-OCRv5 for screen text.
- Microsoft OmniParser v2 for UI elements.
- Qwen2.5-VL-7B-Instruct for screen reasoning.

The project can smoke-test without them, but the demo should be considered real only after
`screen-agent analyze ... --strict-models` works.

## Licensing Note

Review model licenses before any commercial use. Microsoft’s OmniParser README says the V2
`icon_detect` model inherits an AGPL license from YOLO, while its caption models are MIT. That is
acceptable for a local research MVP, but it may be a blocker for a paid hosted product unless the
UI parser is replaced, separately licensed, or isolated in a way counsel approves.

## 1. Base Environment

Use Ubuntu/WSL2 and keep the Hugging Face cache on the Linux filesystem if possible.

```bash
cd "/mnt/c/Users/emran/OneDrive/Documents/New project/screen-agent-mvp"
uv python install 3.11
uv sync --group dev --group models
uv run screen-agent diagnostics
```

Confirm diagnostics shows:

- `cuda_available: true`
- GPU name containing `RTX 5090`
- CUDA 12.8+ or a newer compatible CUDA runtime

## 2. PaddleOCR

The `models` dependency group installs `paddleocr`. If Paddle’s GPU runtime is missing or chooses
CPU, install the Paddle GPU wheel that matches the WSL CUDA runtime, then rerun:

```bash
uv run python -c "from paddleocr import PaddleOCR; print(PaddleOCR)"
```

## 3. Qwen2.5-VL

The default model is:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

First strict run downloads the model weights unless `SCREEN_AGENT_VLM_MODEL` points at a local
folder.

```bash
export SCREEN_AGENT_VLM_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
uv run screen-agent analyze path/to/short-recording.mp4 --out runs/qwen-smoke --strict-models
```

Start with a short recording, because first-run model downloads and warmup are slow.

## 4. OmniParser v2

OmniParser is handled as an external local command because the official project is not packaged as
a simple pip dependency.

Clone and set up OmniParser separately, following Microsoft’s repository instructions:

```bash
cd ~/src
git clone https://github.com/microsoft/OmniParser.git
cd OmniParser
conda create -n omni python==3.12
conda activate omni
pip install -r requirements.txt
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do
  huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done
```

Then expose a small wrapper command that accepts one screenshot path and prints UI elements JSON:

```json
[
  {
    "label": "Pay",
    "kind": "button",
    "x1": 10,
    "y1": 20,
    "x2": 120,
    "y2": 60,
    "confidence": 0.91
  }
]
```

Point the MVP at that wrapper:

```bash
export OMNIPARSER_ROOT="/home/codexssh/OmniParser"
export SCREEN_AGENT_OMNIPARSER_COMMAND="/home/codexssh/screen-agent-mvp/.venv/bin/python /home/codexssh/screen-agent-mvp/scripts/omniparser_json.py"
```

When the command is absent, the MVP uses a heuristic UI parser and records a warning. In strict
mode, the command is required.
