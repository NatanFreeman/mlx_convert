#!/usr/bin/env python3
"""
04_convert_to_mlx.py - Convert Nemotron weights to MLX safetensors format

This script performs the actual weight conversion following the strategy from step 03.

CONVERSION LOGIC (based on parakeet-mlx conversion script):
1. Skip preprocessor weights (audio preprocessing done by librosa)
2. Skip batch norm tracking (not needed for inference)
3. Permute convolution weights to MLX format:
   - 4D: (out, in, h, w) -> (out, h, w, in)
   - 3D: (out, in, len) -> (out, len, in)
4. Rename LSTM weights to MLX format:
   - weight_ih_l* -> *.Wx
   - weight_hh_l* -> *.Wh
   - bias_ih_l* + bias_hh_l* -> *.bias (summed)
5. Save as safetensors

WHY SAFETENSORS:
- Standard format for MLX models
- Fast loading (memory mapped)
- Safe (no pickle vulnerabilities)
- Cross-platform compatible

OUTPUT:
- output/mlx/model.safetensors
- output/mlx/config.json (for parakeet-mlx compatibility)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple

try:
    import torch
    import yaml
    from safetensors.torch import save_file
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError as e:
    print(f"FATAL: Missing required package: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(__file__).parent.parent / "output"
WEIGHTS_FILE = OUTPUT_DIR / "model_weights.ckpt"
CONFIG_FILE = OUTPUT_DIR / "model_config.yaml"
EXTRACTED_DIR = OUTPUT_DIR / "extracted"  # Where 02_extract_weights.py extracts to
MLX_OUTPUT_DIR = OUTPUT_DIR / "mlx"

console = Console()


class ConversionError(Exception):
    """Raised when conversion fails."""
    pass


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_inputs():
    """Validate all required inputs exist."""
    console.print("[bold blue]Validating inputs...[/bold blue]")

    if not WEIGHTS_FILE.exists():
        raise ConversionError(f"Weights file not found: {WEIGHTS_FILE}\nRun steps 01-02 first.")

    if not CONFIG_FILE.exists():
        raise ConversionError(f"Config file not found: {CONFIG_FILE}\nRun steps 01-02 first.")

    if not EXTRACTED_DIR.exists():
        raise ConversionError(f"Extracted directory not found: {EXTRACTED_DIR}\nRun steps 01-02 first.")

    console.print("[green]Inputs validated[/green]")


def load_pytorch_weights() -> Dict[str, torch.Tensor]:
    """Load PyTorch weights."""
    console.print("[bold blue]Loading PyTorch weights...[/bold blue]")

    state_dict = torch.load(WEIGHTS_FILE, map_location='cpu')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    console.print(f"[green]Loaded {len(state_dict)} tensors[/green]")
    return state_dict


def should_skip_key(key: str) -> Tuple[bool, str]:
    """Determine if a key should be skipped and why."""
    # Skip preprocessor (handled by librosa)
    if key.startswith('preprocessor'):
        return True, "preprocessor"

    # Skip batch norm tracking
    if 'num_batches_tracked' in key:
        return True, "batch_norm_tracking"

    return False, ""


def needs_permutation(key: str) -> bool:
    """Determine if a tensor needs dimension permutation.

    Based on actual Nemotron weight structure inspection:
    - encoder.pre_encode.conv.{N}.weight - 2D convolutions for subsampling (4D: [out, in, h, w])
    - encoder.layers.{N}.conv.*.weight - 1D convolutions in conformer blocks (3D: [out, in, len])

    Excludes (these contain 'conv' in their key path but are NOT convolutions):

    1. batch_norm weights:
       - Shape: [1024] (1D)
       - Example: encoder.layers.0.conv.batch_norm.weight
       - Reason: BatchNorm parameters, not conv kernels

    2. norm_conv weights:
       - Shape: [1024] (1D)
       - Example: encoder.layers.0.norm_conv.weight
       - Reason: This is a LayerNorm, NOT a convolution.
         NeMo naming convention: 'norm_X' = "LayerNorm that precedes module X"
         Source: NeMo ConformerLayer defines `self.norm_conv = LayerNorm(d_model)`
         See: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/conformer_modules.py

    Verification (from actual weight inspection):
    - norm_conv.weight: shape [1024] (1D - LayerNorm)
    - conv.depthwise_conv.weight: shape [1024, 1, 9] (3D - actual convolution)
    """
    # Conv weights need permutation
    if 'conv' in key and 'weight' in key:
        # Exclude batch_norm weights (they're 1D BatchNorm parameters)
        if 'batch_norm' in key:
            return False
        # Exclude norm_conv weights (they're 1D LayerNorm parameters, not convolutions)
        # NeMo naming: norm_conv = "the LayerNorm before the conv module"
        if 'norm_conv' in key:
            return False
        return True

    return False


def permute_tensor(tensor: torch.Tensor, key: str) -> torch.Tensor:
    """Permute tensor dimensions for MLX format."""
    ndim = len(tensor.shape)

    if ndim == 4:
        # (out, in, h, w) -> (out, h, w, in)
        return tensor.permute(0, 2, 3, 1)
    elif ndim == 3:
        # (out, in, len) -> (out, len, in)
        return tensor.permute(0, 2, 1)
    else:
        # Unexpected - log warning but don't permute
        console.print(f"[yellow]WARNING: {key} needs permutation but has {ndim}D shape - skipping permute[/yellow]")
        return tensor


def process_lstm_key(key: str) -> Tuple[str, str]:
    """
    Process LSTM weight key for MLX format.

    Based on parakeet-mlx conversion gist, this uses simple string replace:
    - weight_ih_l0 -> 0.Wx (input-to-hidden)
    - weight_hh_l0 -> 0.Wh (hidden-to-hidden)
    - bias_ih_l0 + bias_hh_l0 -> 0.bias (summed)

    Actual keys from Nemotron:
    - decoder.prediction.dec_rnn.lstm.weight_ih_l0
    - decoder.prediction.dec_rnn.lstm.weight_hh_l0
    - decoder.prediction.dec_rnn.lstm.bias_ih_l0
    - decoder.prediction.dec_rnn.lstm.bias_hh_l0

    Returns (new_key, operation) where operation is one of:
    - 'Wx': input-hidden weight
    - 'Wh': hidden-hidden weight
    - 'bias_ih': input-hidden bias (needs summing)
    - 'bias_hh': hidden-hidden bias (needs summing)
    - None: not an LSTM key
    """
    if 'weight_ih_l' in key:
        # decoder.prediction.dec_rnn.lstm.weight_ih_l0 -> decoder.prediction.dec_rnn.lstm.0.Wx
        new_key = key.replace('weight_ih_l', '') + '.Wx'
        return new_key, 'Wx'
    elif 'weight_hh_l' in key:
        # decoder.prediction.dec_rnn.lstm.weight_hh_l0 -> decoder.prediction.dec_rnn.lstm.0.Wh
        new_key = key.replace('weight_hh_l', '') + '.Wh'
        return new_key, 'Wh'
    elif 'bias_ih_l' in key:
        # decoder.prediction.dec_rnn.lstm.bias_ih_l0 -> decoder.prediction.dec_rnn.lstm.0.bias
        new_key = key.replace('bias_ih_l', '') + '.bias'
        return new_key, 'bias_ih'
    elif 'bias_hh_l' in key:
        # decoder.prediction.dec_rnn.lstm.bias_hh_l0 -> decoder.prediction.dec_rnn.lstm.0.bias
        new_key = key.replace('bias_hh_l', '') + '.bias'
        return new_key, 'bias_hh'

    return key, None


def convert_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert PyTorch weights to MLX format."""
    console.print("[bold blue]Converting weights...[/bold blue]")

    new_state_dict = {}
    stats = {
        'skipped': 0,
        'permuted': 0,
        'lstm_renamed': 0,
        'direct_copy': 0,
        'bias_summed': 0,
        'bn_generated': 0,
    }

    # Keep track of which batch norm layers we've already generated buffers for
    bn_layers_processed = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing tensors...", total=len(state_dict))

        for key, tensor in state_dict.items():
            progress.update(task, advance=1, description=f"Processing {key[:50]}...")

            # Check if should skip
            skip, reason = should_skip_key(key)
            if skip:
                stats['skipped'] += 1
                continue

            # Process LSTM keys
            new_key, lstm_op = process_lstm_key(key)

            if lstm_op is not None:
                stats['lstm_renamed'] += 1

                if lstm_op in ['Wx', 'Wh']:
                    new_state_dict[new_key] = tensor
                elif lstm_op == 'bias_ih':
                    if new_key in new_state_dict:
                        new_state_dict[new_key] = new_state_dict[new_key] + tensor
                        stats['bias_summed'] += 1
                    else:
                        new_state_dict[new_key] = tensor
                elif lstm_op == 'bias_hh':
                    if new_key in new_state_dict:
                        new_state_dict[new_key] = new_state_dict[new_key] + tensor
                        stats['bias_summed'] += 1
                    else:
                        new_state_dict[new_key] = tensor
                continue

            # Check if needs permutation
            if needs_permutation(key):
                tensor = permute_tensor(tensor, key)
                stats['permuted'] += 1

            # --- FIX: Generate missing BatchNorm buffers ---
            # The source checkpoint is missing running_mean and running_var, but
            # parakeet-mlx's BatchNorm implementation requires them. We generate
            # them here with standard initial values (zeros for mean, ones for var).
            if ".conv.batch_norm." in key:
                # Get the base path, e.g., 'encoder.layers.0.conv.batch_norm.'
                base_key = key[:key.rfind('.')] + '.'

                # Add running_mean and running_var if we haven't for this layer
                if base_key not in bn_layers_processed:
                    # The size of mean/var is the number of channels, which is
                    # the first (and only) dimension of the weight/bias tensor.
                    size = tensor.shape[0]

                    # Create zero-filled mean and one-filled var tensors
                    running_mean = torch.zeros(size)
                    running_var = torch.ones(size)

                    # Add them to the new state dict
                    new_state_dict[base_key + 'running_mean'] = running_mean
                    new_state_dict[base_key + 'running_var'] = running_var

                    # Mark this layer as processed and update stats
                    bn_layers_processed.add(base_key)
                    stats['bn_generated'] += 2  # 2 new tensors

            # Direct copy with same key
            stats['direct_copy'] += 1
            new_state_dict[key] = tensor

    # Print conversion stats
    console.print()
    console.print("[green]Conversion complete:[/green]")
    console.print(f"  Skipped: {stats['skipped']}")
    console.print(f"  Permuted: {stats['permuted']}")
    console.print(f"  LSTM renamed: {stats['lstm_renamed']}")
    console.print(f"  Bias pairs summed: {stats['bias_summed']}")
    console.print(f"  BatchNorm buffers generated: {stats['bn_generated']} (FIX)")
    console.print(f"  Direct copy: {stats['direct_copy']}")
    console.print(f"  Total output tensors: {len(new_state_dict)}")

    return new_state_dict


def calculate_total_params(state_dict: Dict[str, torch.Tensor]) -> int:
    """Calculate total number of parameters."""
    return sum(t.numel() for t in state_dict.values())


def validate_conversion(original: Dict[str, torch.Tensor], converted: Dict[str, torch.Tensor]):
    """Validate the conversion is reasonable."""
    console.print("[bold blue]Validating conversion...[/bold blue]")

    original_params = calculate_total_params(original)
    converted_params = calculate_total_params(converted)

    console.print(f"  Original params: {original_params:,}")
    console.print(f"  Converted params: {converted_params:,}")

    # Params should be similar (minus preprocessor)
    # Preprocessor is typically a small fraction
    ratio = converted_params / original_params

    if ratio < 0.90:
        raise ValidationError(
            f"Converted model has significantly fewer parameters ({ratio:.1%}).\n"
            f"This may indicate important weights were incorrectly skipped."
        )

    if ratio > 1.01:
        raise ValidationError(
            f"Converted model has more parameters ({ratio:.1%}).\n"
            f"This should not happen - investigate conversion logic."
        )

    console.print(f"  Ratio: {ratio:.2%} [green](OK)[/green]")

    # Verify key components exist
    required_prefixes = ['encoder', 'decoder', 'joint']
    for prefix in required_prefixes:
        matching = [k for k in converted.keys() if k.startswith(prefix)]
        if not matching:
            raise ValidationError(f"No keys with prefix '{prefix}' in converted weights!")
        console.print(f"  {prefix} keys: {len(matching)} [green](OK)[/green]")


def create_mlx_config(nemo_config: dict) -> dict:
    """Create MLX-compatible config from NeMo config.

    CRITICAL: parakeet-mlx expects NeMo config format, NOT a custom schema!

    parakeet-mlx's from_config() (utils.py) does:
    1. Checks config.get("target") to determine model type
    2. Uses dacite.from_dict() to convert config dict to dataclass
    3. Requires: preprocessor, encoder, decoder, joint, decoding sections

    Key findings (2026-01-19):
    - Must have "target" field (NOT "_target_")
    - Vocabulary must be in joint.vocabulary as list of strings
    - Must set causal_downsampling: false (parakeet-mlx doesn't support true)

    NOTE: NeMo config uses "_target_" but parakeet-mlx expects "target" (no underscore)
    """
    console.print("[bold blue]Creating MLX config (NeMo format for parakeet-mlx)...[/bold blue]")

    # Get the sections we need
    preprocessor = nemo_config.get('preprocessor', {})
    encoder = nemo_config.get('encoder', {})
    decoder = nemo_config.get('decoder', {})
    joint = nemo_config.get('joint', {})
    decoding = nemo_config.get('decoding', {})

    # Validate vocabulary exists - this is CRITICAL
    vocabulary = joint.get('vocabulary', [])
    if not vocabulary:
        raise ConversionError(
            "FATAL: No vocabulary found in joint.vocabulary!\n"
            "parakeet-mlx requires vocabulary embedded in config, not separate tokenizer files."
        )
    console.print(f"  Vocabulary size: {len(vocabulary)} tokens")

    # Build config matching parakeet-mlx dataclass expectations
    # See: parakeet_mlx/audio.py PreprocessArgs
    # See: parakeet_mlx/conformer.py ConformerArgs
    # See: parakeet_mlx/rnnt.py PredictArgs, JointArgs
    # See: parakeet_mlx/parakeet.py RNNTDecodingArgs

    mlx_config = {
        # CRITICAL: This determines which model class parakeet-mlx uses
        # Must be "target" not "_target_" (NeMo uses underscore, parakeet-mlx doesn't)
        "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",

        # PreprocessArgs (parakeet_mlx/audio.py)
        "preprocessor": {
            "sample_rate": preprocessor.get('sample_rate', 16000),
            "normalize": preprocessor.get('normalize', 'NA'),
            "window_size": preprocessor.get('window_size', 0.025),
            "window_stride": preprocessor.get('window_stride', 0.01),
            "window": preprocessor.get('window', 'hann'),
            "features": preprocessor.get('features', 128),
            "n_fft": preprocessor.get('n_fft', 512),
            "dither": preprocessor.get('dither', 1e-05),
            "pad_to": preprocessor.get('pad_to', 0),
            "pad_value": preprocessor.get('pad_value', 0.0),
        },

        # ConformerArgs (parakeet_mlx/conformer.py)
        "encoder": {
            "feat_in": encoder.get('feat_in', 128),
            "n_layers": encoder.get('n_layers', 24),
            "d_model": encoder.get('d_model', 1024),
            "n_heads": encoder.get('n_heads', 8),
            "ff_expansion_factor": encoder.get('ff_expansion_factor', 4),
            "subsampling_factor": encoder.get('subsampling_factor', 8),
            "self_attention_model": encoder.get('self_attention_model', 'rel_pos'),
            "subsampling": encoder.get('subsampling', 'dw_striding'),
            "conv_kernel_size": encoder.get('conv_kernel_size', 9),
            "subsampling_conv_channels": encoder.get('subsampling_conv_channels', 256),
            "pos_emb_max_len": encoder.get('pos_emb_max_len', 5000),
            # We enable causal_downsampling because we will patch parakeet-mlx to support it
            "causal_downsampling": encoder.get('causal_downsampling', False),
            "use_bias": encoder.get('use_bias', False),
            "xscaling": encoder.get('xscaling', False),
        },

        # PredictArgs (parakeet_mlx/rnnt.py)
        # Note: parakeet-mlx expects "decoder" to match PredictArgs, not NeMo's RNNTDecoder
        "decoder": {
            "blank_as_pad": decoder.get('blank_as_pad', True),
            "vocab_size": decoder.get('vocab_size', 1024),
            # PredictNetworkArgs nested
            "prednet": {
                "pred_hidden": decoder.get('prednet', {}).get('pred_hidden', 640),
                "pred_rnn_layers": decoder.get('prednet', {}).get('pred_rnn_layers', 2),
            },
        },

        # JointArgs (parakeet_mlx/rnnt.py)
        # CRITICAL: vocabulary must be here as list of strings
        "joint": {
            "num_classes": joint.get('num_classes', 1024),
            "vocabulary": vocabulary,  # List of 1024 token strings
            # JointNetworkArgs nested
            "jointnet": {
                "joint_hidden": joint.get('jointnet', {}).get('joint_hidden', 640),
                "activation": joint.get('jointnet', {}).get('activation', 'relu'),
                "encoder_hidden": joint.get('jointnet', {}).get('encoder_hidden', 1024),
                "pred_hidden": joint.get('jointnet', {}).get('pred_hidden', 640),
            },
        },

        # RNNTDecodingArgs (parakeet_mlx/parakeet.py)
        "decoding": {
            "greedy": decoding.get('greedy', {"max_symbols": 10}),
        },

        # Extra metadata (ignored by dacite but useful for debugging)
        "_conversion_info": {
            "source_model": "nvidia/nemotron-speech-streaming-en-0.6b",
            "original_causal_downsampling": encoder.get('causal_downsampling', True),
            "causal_downsampling_note": "Enabled. Requires patched parakeet-mlx.",
            "conversion_script": "04_convert_to_mlx.py",
        },
    }

    console.print("[green]  Config created with NeMo format for parakeet-mlx[/green]")
    console.print(f"  target: {mlx_config['target']}")
    console.print(f"  causal_downsampling: {mlx_config['encoder']['causal_downsampling']} (workaround)")

    return mlx_config


def save_mlx_model(state_dict: Dict[str, torch.Tensor], config: dict):
    """Save the converted model in MLX format.

    NOTE: We do NOT copy tokenizer files because parakeet-mlx doesn't use them.
    The vocabulary is embedded in config.json under joint.vocabulary.
    See docs/conversion_decisions.md "Tokenizer: Vocabulary Embedded in Config".
    """
    console.print("[bold blue]Saving MLX model...[/bold blue]")

    # Create output directory
    MLX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save weights as safetensors
    weights_path = MLX_OUTPUT_DIR / "model.safetensors"
    save_file(state_dict, str(weights_path))
    console.print(f"[green]Saved weights: {weights_path}[/green]")
    console.print(f"  Size: {weights_path.stat().st_size:,} bytes")

    # Save config (includes vocabulary in joint.vocabulary)
    config_path = MLX_OUTPUT_DIR / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    console.print(f"[green]Saved config: {config_path}[/green]")
    console.print(f"  Vocabulary embedded in config: {len(config['joint']['vocabulary'])} tokens")

    # NOTE: Tokenizer files (*.model, *.vocab) are NOT copied.
    # parakeet-mlx uses joint.vocabulary from config, not SentencePiece files.
    console.print("[dim]  (Tokenizer files not copied - vocabulary is in config.json)[/dim]")

    # Save a README for the output
    readme_path = MLX_OUTPUT_DIR / "README.md"

    readme_content = """# Nemotron Speech Streaming MLX Weights

## Source
- Model: nvidia/nemotron-speech-streaming-en-0.6b
- Architecture: Cache-Aware FastConformer Encoder + RNNT Decoder

## IMPORTANT NOTES

1. **This is RNNT, not TDT**: The decoder is Recurrent Neural Network Transducer.
   Use `ParakeetRNNT` model class, NOT `ParakeetTDT`.

2. **Causal Downsampling Supported (Runtime Patch)**:
   The model uses `causal_downsampling: true`.
   Requires `scripts/parakeet_patch.py` to be applied at runtime.
   
3. **Vocabulary is in config.json**: The tokenizer vocabulary (1024 tokens) is embedded
   in `config.json` under `joint.vocabulary`. Separate tokenizer files are NOT used.

## Usage with parakeet-mlx

```python
from parakeet_mlx import ParakeetRNNT
from scripts.parakeet_patch import apply_patch

# Apply patch BEFORE loading model
apply_patch()

# Load model
model = ParakeetRNNT.from_pretrained("path/to/mlx")

# Batch transcription
text = model.transcribe("audio.wav")

# Streaming transcription
with model.transcribe_stream(context_size=[70, 13]) as stream:
    for chunk in audio_chunks:
        text = stream.transcribe(chunk)
        print(text)
```

## Files
- model.safetensors: Converted weights (~2.4GB)
- config.json: Model configuration with embedded vocabulary

## Conversion
Generated by: scripts/04_convert_to_mlx.py
See: docs/conversion_decisions.md for detailed rationale
"""
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    console.print(f"[green]Saved README: {readme_path}[/green]")


def main():
    """Main entry point."""
    console.print("[bold]=" * 60 + "[/bold]")
    console.print("[bold]Step 4: Convert to MLX Format[/bold]")
    console.print("[bold]=" * 60 + "[/bold]")
    console.print()

    try:
        validate_inputs()

        # NOTE: We no longer find/copy tokenizer files.
        # parakeet-mlx uses vocabulary embedded in config.json, not SentencePiece files.
        # See docs/conversion_decisions.md "Tokenizer: Vocabulary Embedded in Config"

        # Load original weights and config
        pytorch_state = load_pytorch_weights()

        with open(CONFIG_FILE, 'r') as f:
            nemo_config = yaml.safe_load(f)

        # Convert weights
        mlx_state = convert_weights(pytorch_state)

        # Validate conversion
        validate_conversion(pytorch_state, mlx_state)

        # Create MLX config (includes vocabulary from NeMo config)
        mlx_config = create_mlx_config(nemo_config)

        # Save model (vocabulary embedded in config, no separate tokenizer files)
        save_mlx_model(mlx_state, mlx_config)

        console.print()
        console.print("[bold green]SUCCESS: Model converted to MLX format[/bold green]")
        console.print()
        console.print(f"Output directory: {MLX_OUTPUT_DIR}")
        console.print()
        console.print("[green]IMPORTANT: causal_downsampling enabled via runtime patch[/green]")
        console.print("[green]Full weights restored (no pruning needed)[/green]")
        console.print()
        console.print("Next step: [bold green]uv run scripts/05_test_inference.py[/bold green] (or pip: python scripts/05_test_inference.py)")
        console.print()
        console.print("[yellow]NOTE: Step 5 requires Apple Silicon Mac with MLX installed[/yellow]")

    except (ConversionError, ValidationError) as e:
        console.print()
        console.print(f"[bold red]FATAL ERROR[/bold red]")
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    except Exception as e:
        console.print()
        console.print(f"[bold red]UNEXPECTED ERROR[/bold red]")
        console.print(f"[red]{type(e).__name__}: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
