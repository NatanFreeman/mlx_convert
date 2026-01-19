#!/usr/bin/env python3
"""
03_analyze_architecture.py - Deep analysis of weight structure for MLX conversion

This script performs detailed analysis to determine:
1. Exact mapping between Nemotron weights and parakeet-mlx expected format
2. Any architectural differences that require special handling
3. Whether a direct conversion is possible or if custom MLX code is needed

WHY THIS STEP IS CRITICAL:
- The parakeet-mlx conversion script was designed for Parakeet-TDT
- Nemotron uses RNNT decoder (different from TDT)
- Nemotron has cache-aware streaming (may have additional weights)
- Silent mismatches cause quality degradation

OUTPUTS:
- Detailed weight mapping analysis
- Compatibility report
- Conversion strategy recommendation
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

try:
    import torch
    import yaml
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
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

console = Console()


class AnalysisError(Exception):
    """Raised when analysis fails."""
    pass


class IncompatibilityError(Exception):
    """Raised when architecture is incompatible with target format."""
    pass


# ============================================================
# PARAKEET-MLX EXPECTED WEIGHT PATTERNS
# Based on: https://gist.github.com/senstella/77178bb5d6ec67bf8c54705a5f490bed
# ============================================================

# The conversion script transforms these patterns:
# NOTE: Based on actual inspection of nvidia/nemotron-speech-streaming-en-0.6b weights
PARAKEET_MLX_PATTERNS = {
    # Preprocessor weights are SKIPPED (handled by librosa)
    'skip_prefixes': ['preprocessor'],

    # Batch norm tracking is SKIPPED (if present - not found in actual weights)
    'skip_contains': ['num_batches_tracked'],

    # Convolution weights need dimension permutation
    # Based on actual Nemotron weight inspection:
    # - encoder.pre_encode.conv.{N}.weight - 2D convolutions for subsampling (4D tensors)
    # - encoder.layers.{N}.conv.*.weight - 1D convolutions in conformer blocks (3D tensors)
    # 4D: (out, in, h, w) -> (out, h, w, in)  [permute (0,2,3,1)]
    # 3D: (out, in, len) -> (out, len, in)    [permute (0,2,1)]
    'permute_patterns': ['conv'],  # Applied only to .weight keys
    'permute_exclude': [
        'batch_norm',  # BatchNorm params, shape [1024], not conv kernels
        'norm_conv',   # LayerNorm params, shape [1024], NOT a convolution despite the name!
                       # NeMo naming convention: 'norm_X' = "LayerNorm that precedes module X"
                       # Source: NeMo ConformerLayer defines `self.norm_conv = LayerNorm(d_model)`
                       # See: github.com/NVIDIA/NeMo/.../conformer_modules.py
    ],

    # LSTM weight renaming (decoder.prediction.dec_rnn.lstm.*)
    # weight_ih_l* -> *.Wx
    # weight_hh_l* -> *.Wh
    # bias_ih_l* + bias_hh_l* -> *.bias (summed)
    'lstm_patterns': ['weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l'],
}


def load_weights():
    """Load the extracted weights."""
    console.print("[bold blue]Loading weights...[/bold blue]")

    if not WEIGHTS_FILE.exists():
        raise AnalysisError(f"Weights file not found: {WEIGHTS_FILE}\nRun step 02 first.")

    state_dict = torch.load(WEIGHTS_FILE, map_location='cpu')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    console.print(f"[green]Loaded {len(state_dict)} weight tensors[/green]")
    return state_dict


def load_config():
    """Load the model config."""
    console.print("[bold blue]Loading config...[/bold blue]")

    if not CONFIG_FILE.exists():
        raise AnalysisError(f"Config file not found: {CONFIG_FILE}\nRun step 02 first.")

    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    return config


def categorize_weights(state_dict):
    """Categorize weights by how they should be handled for MLX conversion."""
    console.print("[bold blue]Categorizing weights for conversion...[/bold blue]")

    categories = {
        'skip': [],           # Will be skipped entirely
        'permute_4d': [],     # Need 4D permutation
        'permute_3d': [],     # Need 3D permutation
        'lstm_rename': [],    # Need LSTM weight renaming
        'direct_copy': [],    # Copy as-is
    }

    for key, tensor in state_dict.items():
        # Check skip patterns
        if any(key.startswith(prefix) for prefix in PARAKEET_MLX_PATTERNS['skip_prefixes']):
            categories['skip'].append((key, tensor.shape, 'preprocessor'))
            continue

        if any(pattern in key for pattern in PARAKEET_MLX_PATTERNS['skip_contains']):
            categories['skip'].append((key, tensor.shape, 'batch_norm_tracking'))
            continue

        # Check LSTM patterns
        if any(pattern in key for pattern in PARAKEET_MLX_PATTERNS['lstm_patterns']):
            categories['lstm_rename'].append((key, tensor.shape))
            continue

        # Check permutation patterns
        # Must match permute_patterns AND NOT match permute_exclude AND be a .weight key
        needs_permute = (
            any(pattern in key for pattern in PARAKEET_MLX_PATTERNS['permute_patterns']) and
            not any(excl in key for excl in PARAKEET_MLX_PATTERNS['permute_exclude']) and
            key.endswith('.weight')
        )

        if needs_permute:
            if len(tensor.shape) == 4:
                categories['permute_4d'].append((key, tensor.shape))
            elif len(tensor.shape) == 3:
                categories['permute_3d'].append((key, tensor.shape))
            else:
                # Unexpected dimension for permutation pattern
                console.print(f"[yellow]WARNING: {key} matches permute pattern but has {len(tensor.shape)}D shape[/yellow]")
                categories['direct_copy'].append((key, tensor.shape))
        else:
            categories['direct_copy'].append((key, tensor.shape))

    # Print summary
    table = Table(title="Weight Categorization Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Description", style="dim")

    table.add_row("skip", str(len(categories['skip'])), "Preprocessor/batch norm - not needed")
    table.add_row("permute_4d", str(len(categories['permute_4d'])), "4D convolutions - need permutation")
    table.add_row("permute_3d", str(len(categories['permute_3d'])), "3D convolutions - need permutation")
    table.add_row("lstm_rename", str(len(categories['lstm_rename'])), "LSTM weights - need renaming")
    table.add_row("direct_copy", str(len(categories['direct_copy'])), "Copy as-is")

    console.print(table)

    return categories


def analyze_encoder_structure(state_dict):
    """Analyze encoder structure for cache-aware specific components."""
    console.print("[bold blue]Analyzing encoder structure...[/bold blue]")

    encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder')]

    # Look for cache-aware specific patterns
    cache_patterns = ['cache', 'stream', 'left_context', 'right_context']
    cache_related = [k for k in encoder_keys if any(p in k.lower() for p in cache_patterns)]

    if cache_related:
        console.print(f"[yellow]Found {len(cache_related)} cache-related weight keys:[/yellow]")
        for key in cache_related[:10]:
            console.print(f"  - {key}")
        if len(cache_related) > 10:
            console.print(f"  ... and {len(cache_related) - 10} more")
    else:
        console.print("[green]No cache-specific weights found (caching is runtime behavior)[/green]")

    # Analyze layer structure
    layer_pattern = defaultdict(list)
    for key in encoder_keys:
        # Extract layer number if present
        parts = key.split('.')
        for i, part in enumerate(parts):
            if part.isdigit() and i > 0 and parts[i-1] in ['layers', 'layer']:
                layer_pattern[int(part)].append(key)
                break

    if layer_pattern:
        console.print(f"[green]Found {len(layer_pattern)} encoder layers[/green]")
        # Show first layer structure as sample
        if 0 in layer_pattern:
            console.print("Sample layer 0 structure:")
            for key in sorted(layer_pattern[0])[:5]:
                console.print(f"  - {key}")

    return {
        'total_encoder_keys': len(encoder_keys),
        'cache_related_keys': len(cache_related),
        'num_layers': len(layer_pattern),
    }


def analyze_decoder_structure(state_dict):
    """Analyze decoder structure to confirm RNNT (not TDT)."""
    console.print("[bold blue]Analyzing decoder structure...[/bold blue]")

    decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder')]
    joint_keys = [k for k in state_dict.keys() if k.startswith('joint')]

    # TDT-specific patterns (should NOT be present)
    tdt_patterns = ['duration', 'tdt', 'time_delay']
    tdt_found = [k for k in state_dict.keys() if any(p in k.lower() for p in tdt_patterns)]

    if tdt_found:
        console.print("[bold red]CRITICAL: Found TDT-related weights![/bold red]")
        for key in tdt_found[:5]:
            console.print(f"  - {key}")
        raise IncompatibilityError(
            "This model appears to have TDT decoder components.\n"
            "This is NOT the Nemotron Speech Streaming RNNT model!\n"
            "Found keys: " + str(tdt_found[:5])
        )

    console.print(f"[green]Decoder keys: {len(decoder_keys)}[/green]")
    console.print(f"[green]Joint network keys: {len(joint_keys)}[/green]")

    if len(joint_keys) == 0:
        console.print("[yellow]WARNING: No joint network found - unusual for RNNT[/yellow]")

    # Show decoder structure
    console.print("Decoder key samples:")
    for key in sorted(decoder_keys)[:5]:
        console.print(f"  - {key}: {list(state_dict[key].shape)}")

    console.print("Joint key samples:")
    for key in sorted(joint_keys)[:5]:
        console.print(f"  - {key}: {list(state_dict[key].shape)}")

    return {
        'decoder_keys': len(decoder_keys),
        'joint_keys': len(joint_keys),
        'tdt_found': len(tdt_found),
    }


def generate_conversion_strategy(categories, encoder_info, decoder_info, config):
    """Generate the conversion strategy and compatibility assessment."""
    console.print("[bold blue]Generating conversion strategy...[/bold blue]")

    # Assess compatibility with parakeet-mlx
    compatibility_issues = []
    recommendations = []

    # Check if RNNT is supported by parakeet-mlx
    # Based on research, parakeet-mlx supports: ParakeetTDT, ParakeetRNNT, ParakeetCTC, ParakeetTDTCTC
    recommendations.append("parakeet-mlx supports RNNT - use ParakeetRNNT model class")

    # Check encoder compatibility
    if encoder_info['num_layers'] != 24:
        compatibility_issues.append(
            f"Encoder has {encoder_info['num_layers']} layers, expected 24"
        )

    # Cache-aware streaming is runtime behavior, not weight-dependent
    if encoder_info['cache_related_keys'] == 0:
        recommendations.append(
            "Cache-aware streaming is runtime configuration, not in weights. "
            "Use transcribe_stream() with appropriate context_size for streaming."
        )

    # Check LSTM handling
    if len(categories['lstm_rename']) > 0:
        recommendations.append(
            f"Found {len(categories['lstm_rename'])} LSTM weights that need renaming for MLX"
        )

    # Generate strategy
    strategy = {
        'compatible': len(compatibility_issues) == 0,
        'issues': compatibility_issues,
        'recommendations': recommendations,
        'weight_handling': {
            'skip': len(categories['skip']),
            'permute': len(categories['permute_4d']) + len(categories['permute_3d']),
            'lstm_rename': len(categories['lstm_rename']),
            'direct_copy': len(categories['direct_copy']),
        }
    }

    # Print strategy
    panel_content = []

    if strategy['compatible']:
        panel_content.append("[bold green]COMPATIBLE with parakeet-mlx[/bold green]")
    else:
        panel_content.append("[bold red]COMPATIBILITY ISSUES FOUND[/bold red]")
        for issue in compatibility_issues:
            panel_content.append(f"[red]  - {issue}[/red]")

    panel_content.append("")
    panel_content.append("[bold]Conversion Steps:[/bold]")
    panel_content.append(f"  1. Skip {strategy['weight_handling']['skip']} preprocessor weights")
    panel_content.append(f"  2. Permute {strategy['weight_handling']['permute']} convolution weights")
    panel_content.append(f"  3. Rename {strategy['weight_handling']['lstm_rename']} LSTM weights")
    panel_content.append(f"  4. Direct copy {strategy['weight_handling']['direct_copy']} weights")
    panel_content.append(f"  5. Save as safetensors format")

    panel_content.append("")
    panel_content.append("[bold]Recommendations:[/bold]")
    for rec in recommendations:
        panel_content.append(f"  - {rec}")

    console.print(Panel("\n".join(panel_content), title="Conversion Strategy"))

    # Save strategy to file
    strategy_path = OUTPUT_DIR / "conversion_strategy.yaml"
    with open(strategy_path, 'w') as f:
        yaml.dump(strategy, f, default_flow_style=False)
    console.print(f"[green]Saved strategy to: {strategy_path}[/green]")

    return strategy


def main():
    """Main entry point."""
    console.print("[bold]=" * 60 + "[/bold]")
    console.print("[bold]Step 3: Architecture Analysis for MLX Conversion[/bold]")
    console.print("[bold]=" * 60 + "[/bold]")
    console.print()

    try:
        state_dict = load_weights()
        config = load_config()

        categories = categorize_weights(state_dict)
        encoder_info = analyze_encoder_structure(state_dict)
        decoder_info = analyze_decoder_structure(state_dict)

        strategy = generate_conversion_strategy(categories, encoder_info, decoder_info, config)

        console.print()
        if strategy['compatible']:
            console.print("[bold green]SUCCESS: Architecture analysis complete[/bold green]")
            console.print()
            console.print(f"Next step: [bold green]uv run scripts/04_convert_to_mlx.py[/bold green] (or pip: python scripts/04_convert_to_mlx.py)")
        else:
            console.print("[bold red]INCOMPATIBILITY DETECTED[/bold red]")
            console.print("Review the issues above before proceeding.")
            sys.exit(1)

    except (AnalysisError, IncompatibilityError) as e:
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
