# Nemotron Speech Streaming to MLX Conversion

## Overview

This project converts NVIDIA's `nemotron-speech-streaming-en-0.6b` model to MLX format for inference on Apple Silicon.

**Source Model**: [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)

## Critical Architecture Notes

### What This Model Is

The Nemotron Speech Streaming model is a **Cache-Aware FastConformer encoder with RNNT decoder**:

- **Encoder**: 24-layer Cache-Aware FastConformer with 8x depthwise-separable convolutional downsampling
- **Decoder**: RNNT (Recurrent Neural Network Transducer) - NOT TDT
- **Parameters**: 600M
- **Input**: 16kHz mono audio, minimum 80ms
- **Output**: English text with punctuation and capitalization

### Why This Is NOT Parakeet-TDT

| Attribute | Nemotron-Speech-Streaming | Parakeet-TDT-0.6b-v3 |
|-----------|---------------------------|----------------------|
| Encoder | Cache-Aware FastConformer | FastConformer |
| Decoder | RNNT | TDT |
| Streaming | Native cache-based | Chunked inference |
| Inference | Stateful (caches activations) | Stateless |

**WARNING**: Some repositories on HuggingFace (e.g., `animaslabs/nemotron-speech-streaming-en-0.6b-mlx`) are mislabeled - they contain Parakeet-TDT weights renamed to look like Nemotron. This causes silent quality degradation.

### Cache-Aware Architecture (Paper: arXiv 2312.17279)

The cache-aware mechanism:
1. Maintains caches for all encoder self-attention and convolution layers
2. Reuses hidden states across streaming steps
3. Eliminates redundant computations between chunks
4. Each audio frame is processed exactly once

This is fundamentally different from buffered streaming which re-processes overlapping windows.

---

## Critical parakeet-mlx Compatibility Notes

### 1. Config Format Must Match NeMo Format

parakeet-mlx expects config.json in **NeMo format**, not a custom schema.

**Source**: [`parakeet_mlx/utils.py` lines 22-52](https://github.com/senstella/parakeet-mlx/blob/de9ead8/parakeet_mlx/utils.py#L22-L52) (commit de9ead8, version 0.5.0):

```python
def from_config(config: dict) -> BaseParakeet:
    # ...
    elif (
        config.get("target")  # Must be "target", not "_target_"
        == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
        and config.get("model_defaults", {}).get("tdt_durations") is None
    ):
        cfg = from_dict(ParakeetRNNTArgs, config)  # Uses dacite to map config to dataclass
        model = ParakeetRNNT(cfg)
```

The config must have: `target`, `preprocessor`, `encoder`, `decoder`, `joint`, `decoding` sections.

### 2. Vocabulary Is Embedded in Config, NOT Separate Files

parakeet-mlx does **NOT** use SentencePiece tokenizer files. The vocabulary is embedded directly in `config.json`.

**Sources**:
- [`parakeet_mlx/rnnt.py` line 32](https://github.com/senstella/parakeet-mlx/blob/de9ead8/parakeet_mlx/rnnt.py#L32): `vocabulary: list[str]` in JointArgs dataclass
- [`parakeet_mlx/tokenizer.py` lines 2-3](https://github.com/senstella/parakeet-mlx/blob/de9ead8/parakeet_mlx/tokenizer.py#L2-L3): decode function

```python
# parakeet_mlx/tokenizer.py - just looks up tokens by index:
def decode(tokens: list[int], vocabulary: list[str]):
    return "".join([vocabulary[token].replace("▁", " ") for token in tokens])
```

The 1024-token vocabulary must be in `config["joint"]["vocabulary"]`.

### 3. Causal Downsampling Not Supported (LIMITATION)

The Nemotron model uses `causal_downsampling: true` for streaming. **parakeet-mlx doesn't support this**.

**Source**: [`parakeet_mlx/conformer.py` lines 355-362](https://github.com/senstella/parakeet-mlx/blob/de9ead8/parakeet_mlx/conformer.py#L355-L362) (commit de9ead8, version 0.5.0):

```python
if args.subsampling_factor > 1:
    if args.subsampling == "dw_striding" and args.causal_downsampling is False:
        self.pre_encode = DwStridingSubsampling(args)
    else:
        self.pre_encode = nn.Identity()
        raise NotImplementedError(
            "Other subsampling haven't been implemented yet!"
        )
```

**Workaround**: We set `causal_downsampling: false`. The weights are compatible (same shapes), but streaming inference may have boundary artifacts.

---

## Two-Stage Workflow

This conversion has **two distinct stages** with different platform requirements:

### Stage 1: Extraction & Conversion (Steps 01-04)
**Platform**: Any machine with Python and PyTorch (Linux, macOS, Windows)
**Purpose**: Download the .nemo file, extract weights, analyze architecture, convert to safetensors
**Output**: `output/mlx/` directory containing converted model

### Stage 2: MLX Testing (Step 05)
**Platform**: Apple Silicon Mac only (M1/M2/M3/M4/M5)
**Purpose**: Load converted model in MLX and verify inference produces correct transcriptions
**Why this matters**: Conversion can succeed but produce wrong results if weight mappings are incorrect

---

## Project Structure

```
mlx_convert/
├── README.md                       # This file
├── requirements.txt                # Stage 1 dependencies (PyTorch, any platform)
├── requirements-mlx.txt            # Stage 2 dependencies (MLX, Apple Silicon only)
├── scripts/
│   ├── 01_download_model.py        # Download .nemo from HuggingFace
│   ├── 02_extract_weights.py       # Extract weights and config from .nemo
│   ├── 03_analyze_architecture.py  # Analyze weight structure for conversion
│   ├── 04_convert_to_mlx.py        # Convert to MLX safetensors format
│   └── 05_test_inference.py        # Test converted model (Apple Silicon only)
├── docs/
│   └── conversion_decisions.md     # Detailed rationale for every decision
└── output/                         # Generated (disposable, can be regenerated)
    ├── nemotron-speech-streaming-en-0.6b.nemo  # Downloaded model
    ├── extracted/                  # Extracted .nemo contents
    ├── model_weights.ckpt          # Extracted PyTorch weights
    ├── model_config.yaml           # Extracted NeMo config
    ├── weight_keys.txt             # All weight keys for inspection
    ├── conversion_strategy.yaml    # Generated conversion plan
    └── mlx/                        # Final MLX model (copy this to Mac)
        ├── model.safetensors       # Converted weights (~2.4GB)
        ├── config.json             # Model config (includes vocabulary)
        └── README.md               # Usage instructions
```

---

## Usage

### Stage 1: Extraction & Conversion (Any Platform)

#### uv (Recommended)

[`uv`](https://github.com/astral-sh/uv) is an extremely fast Python package manager.

```bash
cd /path/to/mlx_convert

# Run conversion scripts (automatically creates environment and installs dependencies)
uv run scripts/01_download_model.py
uv run scripts/02_extract_weights.py
uv run scripts/03_analyze_architecture.py
uv run scripts/04_convert_to_mlx.py
```

#### pip (Legacy)

```bash
cd /path/to/mlx_convert

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Stage 1 dependencies
pip install -r requirements.txt

# Run Steps 01-04 in order
python scripts/01_download_model.py    # Downloads ~2.5GB .nemo file
python scripts/02_extract_weights.py   # Extracts weights and config
python scripts/03_analyze_architecture.py  # Analyzes structure, generates strategy
python scripts/04_convert_to_mlx.py    # Converts to MLX format

# Output is in output/mlx/
ls -la output/mlx/
```

### Stage 2: MLX Testing (Apple Silicon Mac Only)

If you ran Stage 1 on a different machine, first copy the output:

```bash
# On the machine that ran Stage 1:
# Copy the entire output/mlx/ directory to your Mac
# Option A: scp
scp -r output/mlx/ user@your-mac:/path/to/mlx_convert/output/

# Option B: rsync (preserves permissions)
rsync -avz output/mlx/ user@your-mac:/path/to/mlx_convert/output/mlx/

# Option C: Archive and transfer
tar -czf mlx_model.tar.gz output/mlx/
# Then transfer mlx_model.tar.gz to Mac and extract
```

On your Apple Silicon Mac:

#### uv (Recommended)

```bash
cd /path/to/mlx_convert

# Install the 'mlx' optional dependency group
uv sync --extra mlx

# Run inference test
uv run scripts/05_test_inference.py
```

#### pip (Legacy)

```bash
cd /path/to/mlx_convert

# Create virtual environment (or reuse existing)
python3 -m venv venv
source venv/bin/activate

# Install Stage 1 dependencies (needed for script imports)
pip install -r requirements.txt

# Install Stage 2 dependencies (MLX - Apple Silicon only)
pip install -r requirements-mlx.txt

# Run inference test
python scripts/05_test_inference.py
```

### What Step 05 Tests

Step 05 performs these critical validations:

1. **Model Loading**: Verifies MLX can load the safetensors file
2. **Weight Shape Verification**: Confirms all weight shapes match expected architecture
3. **Config Validation**: Verifies vocabulary is present in config.json
4. **Inference Test**: Runs actual transcription on test audio
5. **Output Validation**: Checks that transcription produces text (not empty/garbage)

If Step 05 fails, the conversion is **not valid** even if Steps 01-04 succeeded.

**Note**: Due to the `causal_downsampling` limitation, streaming tests may show boundary artifacts.

---

## Error Handling Philosophy

Every script is designed to **fail loudly** rather than silently:

- **No fallbacks**: If something unexpected happens, the script raises an exception
- **Strict validation**: Each step validates its inputs and outputs
- **Assertions everywhere**: Expected conditions are asserted, not assumed
- **No silent degradation**: We never produce "close enough" results

Example: If a weight tensor has an unexpected shape, the script crashes with details rather than silently truncating or padding.

---

## Dependencies

### requirements.txt (Stage 1 - Any Platform)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.1.0 | Load .nemo/.ckpt files, tensor operations |
| safetensors | 0.7.0 | Save weights in MLX-compatible format |
| PyYAML | ≥6.0.1 | Parse NeMo config files |
| rich | ≥13.0.0 | Formatted console output |

### requirements-mlx.txt (Stage 2 - Apple Silicon Only)

| Package | Version | Purpose |
|---------|---------|---------|
| mlx | 0.30.3 | Apple's ML framework for Apple Silicon |
| mlx-audio | 0.2.10 | Audio processing on MLX |
| parakeet-mlx | 0.4.1 | NVIDIA Parakeet model implementations for MLX |

---

## References

- Paper: [Stateful Conformer with Cache-based Inference](https://arxiv.org/abs/2312.17279)
- Paper: [Fast Conformer with Linearly Scalable Attention](https://arxiv.org/abs/2305.05084)
- [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [parakeet-mlx](https://github.com/senstella/parakeet-mlx) (reference implementation for similar models)
- [Original conversion gist](https://gist.github.com/senstella/77178bb5d6ec67bf8c54705a5f490bed) (parakeet-mlx conversion reference)
