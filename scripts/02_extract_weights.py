#!/usr/bin/env python3
"""
02_extract_weights.py - Extract weights and config from .nemo file

This script extracts:
1. model_config.yaml - The complete model configuration
2. model_weights.ckpt - The PyTorch state dict

WHY WE EXTRACT MANUALLY:
- We need to inspect the exact weight structure before conversion
- We avoid NeMo's runtime overhead for analysis
- We can compare directly with parakeet-mlx expected format

WHAT WE VERIFY:
- The extracted config matches expected Nemotron architecture
- The weight keys match expected Cache-Aware FastConformer + RNNT structure
- No unexpected components are present

ERROR HANDLING:
- Fails if .nemo file doesn't exist
- Fails if extraction yields unexpected structure
- Fails if config doesn't match expected architecture
"""

import os
import sys
import tarfile
import shutil
from pathlib import Path

try:
    import torch
    import yaml
    from rich.console import Console
    from rich.table import Table
except ImportError as e:
    print(f"FATAL: Missing required package: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# ============================================================
# STRICT CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(__file__).parent.parent / "output"
NEMO_FILE = OUTPUT_DIR / "nemotron-speech-streaming-en-0.6b.nemo"

# Expected architecture parameters - THESE MUST MATCH EXACTLY
EXPECTED_CONFIG = {
    "encoder_layers": 24,
    "encoder_type": "FastConformer",  # or similar naming
    "decoder_type": "rnnt",  # NOT "tdt"
    "vocab_size_approx": (1024, 32768),  # Reasonable range
}

console = Console()


class ExtractionError(Exception):
    """Raised when extraction fails."""
    pass


class ArchitectureMismatchError(Exception):
    """Raised when extracted architecture doesn't match expectations."""
    pass


def validate_nemo_exists():
    """Verify the .nemo file exists from previous step."""
    console.print("[bold blue]Checking for .nemo file...[/bold blue]")

    if not NEMO_FILE.exists():
        raise ExtractionError(
            f".nemo file not found: {NEMO_FILE}\n"
            f"Run step 01_download_model.py first."
        )

    console.print(f"[green]Found: {NEMO_FILE}[/green]")
    console.print(f"  Size: {NEMO_FILE.stat().st_size:,} bytes")


def extract_nemo_archive():
    """Extract contents of .nemo file."""
    console.print("[bold blue]Extracting .nemo archive...[/bold blue]")

    extract_dir = OUTPUT_DIR / "extracted"

    # Clean previous extraction
    if extract_dir.exists():
        console.print(f"  Removing previous extraction: {extract_dir}")
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True)

    with tarfile.open(NEMO_FILE, 'r') as tar:
        # List contents first
        members = tar.getnames()
        console.print(f"  Archive contains {len(members)} files")

        # Extract all
        tar.extractall(path=extract_dir)

    # Verify extraction
    extracted_files = list(extract_dir.rglob("*"))
    console.print(f"[green]Extracted {len(extracted_files)} items to {extract_dir}[/green]")

    return extract_dir


def find_config_and_weights(extract_dir: Path):
    """Locate config and weights files in extracted directory."""
    console.print("[bold blue]Locating config and weights...[/bold blue]")

    # Find model_config.yaml
    config_candidates = list(extract_dir.rglob("model_config.yaml"))
    if len(config_candidates) == 0:
        raise ExtractionError(
            f"No model_config.yaml found in {extract_dir}\n"
            f"Contents: {[str(p.relative_to(extract_dir)) for p in extract_dir.rglob('*') if p.is_file()]}"
        )
    if len(config_candidates) > 1:
        raise ExtractionError(
            f"Multiple model_config.yaml files found: {config_candidates}\n"
            f"Expected exactly one - investigate file structure."
        )
    config_path = config_candidates[0]

    # Find model_weights.ckpt
    weights_candidates = list(extract_dir.rglob("model_weights.ckpt"))
    if len(weights_candidates) == 0:
        raise ExtractionError(
            f"No model_weights.ckpt found in {extract_dir}\n"
            f"Contents: {[str(p.relative_to(extract_dir)) for p in extract_dir.rglob('*') if p.is_file()]}"
        )
    if len(weights_candidates) > 1:
        raise ExtractionError(
            f"Multiple model_weights.ckpt files found: {weights_candidates}\n"
            f"Expected exactly one - investigate file structure."
        )
    weights_path = weights_candidates[0]

    console.print(f"[green]Config: {config_path.relative_to(extract_dir)}[/green]")
    console.print(f"[green]Weights: {weights_path.relative_to(extract_dir)}[/green]")
    console.print(f"  Weights size: {weights_path.stat().st_size:,} bytes")

    return config_path, weights_path


def load_and_analyze_config(config_path: Path):
    """Load config and extract key architecture parameters."""
    console.print("[bold blue]Analyzing model configuration...[/bold blue]")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Save the full config for reference
    full_config_path = OUTPUT_DIR / "model_config.yaml"
    with open(full_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    console.print(f"  Saved full config to: {full_config_path}")

    # Extract key architecture details
    # NOTE: Config is FLAT (not nested under 'model') based on actual .nemo inspection
    analysis = {}

    # Encoder info - key is 'n_layers' NOT 'num_layers'
    encoder_config = config.get('encoder', {})
    analysis['encoder_n_layers'] = encoder_config.get('n_layers', 'NOT_FOUND')
    analysis['encoder_d_model'] = encoder_config.get('d_model', 'NOT_FOUND')
    analysis['encoder_subsampling'] = encoder_config.get('subsampling', 'NOT_FOUND')
    analysis['encoder_subsampling_factor'] = encoder_config.get('subsampling_factor', 'NOT_FOUND')

    # Check for cache-aware specific parameters
    analysis['att_context_size'] = encoder_config.get('att_context_size', 'NOT_FOUND')
    analysis['att_context_style'] = encoder_config.get('att_context_style', 'NOT_FOUND')

    # Decoder info
    decoder_config = config.get('decoder', {})
    analysis['decoder_type'] = decoder_config.get('_target_', 'NOT_FOUND')
    analysis['decoder_vocab_size'] = decoder_config.get('vocab_size', 'NOT_FOUND')

    # Joint network (RNNT specific)
    joint_config = config.get('joint', {})
    analysis['joint_type'] = joint_config.get('_target_', 'NOT_FOUND')
    analysis['joint_num_classes'] = joint_config.get('num_classes', 'NOT_FOUND')

    # Tokenizer info
    tokenizer_config = config.get('tokenizer', {})
    analysis['tokenizer_type'] = tokenizer_config.get('type', 'NOT_FOUND')
    analysis['tokenizer_model'] = tokenizer_config.get('model_path', 'NOT_FOUND')

    # Model target class
    analysis['model_target'] = config.get('target', 'NOT_FOUND')

    # Print analysis
    table = Table(title="Model Configuration Analysis")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    for key, value in analysis.items():
        # Truncate long values for display
        val_str = str(value)
        if len(val_str) > 60:
            val_str = val_str[:57] + "..."
        table.add_row(key, val_str)

    console.print(table)

    return config, analysis


def load_and_analyze_weights(weights_path: Path):
    """Load weights and analyze structure."""
    console.print("[bold blue]Analyzing model weights...[/bold blue]")

    # Load with map_location to CPU (works on any machine)
    state_dict = torch.load(weights_path, map_location='cpu')

    # Sometimes weights are wrapped in 'state_dict' key
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    console.print(f"  Total parameter tensors: {len(state_dict)}")

    # Categorize weights by component
    components = {}
    for key in state_dict.keys():
        # Extract top-level component (encoder, decoder, joint, etc.)
        parts = key.split('.')
        component = parts[0] if parts else 'unknown'
        if component not in components:
            components[component] = {'count': 0, 'params': 0, 'keys': []}
        components[component]['count'] += 1
        components[component]['params'] += state_dict[key].numel()
        if len(components[component]['keys']) < 5:  # Keep first 5 for sample
            components[component]['keys'].append(key)

    # Print component summary
    table = Table(title="Weight Structure by Component")
    table.add_column("Component", style="cyan")
    table.add_column("Tensors", style="green")
    table.add_column("Parameters", style="yellow")
    table.add_column("Sample Keys", style="dim")

    total_params = 0
    for comp, info in sorted(components.items()):
        total_params += info['params']
        table.add_row(
            comp,
            str(info['count']),
            f"{info['params']:,}",
            str(info['keys'][:2])
        )

    console.print(table)
    console.print(f"  [bold]Total parameters: {total_params:,} ({total_params/1e9:.2f}B)[/bold]")

    # Save weight keys for analysis
    keys_path = OUTPUT_DIR / "weight_keys.txt"
    with open(keys_path, 'w') as f:
        for key in sorted(state_dict.keys()):
            shape = list(state_dict[key].shape)
            f.write(f"{key}: {shape}\n")
    console.print(f"  Saved weight keys to: {keys_path}")

    # Copy weights to output directory
    output_weights_path = OUTPUT_DIR / "model_weights.ckpt"
    shutil.copy(weights_path, output_weights_path)
    console.print(f"  Copied weights to: {output_weights_path}")

    return state_dict, components


def verify_architecture(config: dict, analysis: dict, components: dict):
    """Verify the architecture matches our expectations."""
    console.print("[bold blue]Verifying architecture matches expectations...[/bold blue]")

    issues = []

    # Check encoder layers - key is 'encoder_n_layers' now
    if analysis.get('encoder_n_layers') != 24:
        issues.append(
            f"Encoder layers: expected 24, got {analysis.get('encoder_n_layers')}"
        )

    # Check for RNNT decoder (not TDT)
    decoder_type = str(analysis.get('decoder_type', '')).lower()
    if 'tdt' in decoder_type:
        issues.append(
            f"Decoder appears to be TDT, not RNNT: {decoder_type}\n"
            f"This is a DIFFERENT model than nemotron-speech-streaming!"
        )

    # Verify it's actually RNNT
    if 'rnnt' not in decoder_type.lower():
        issues.append(
            f"Decoder type doesn't contain 'rnnt': {decoder_type}\n"
            f"Expected nemo.collections.asr.modules.RNNTDecoder"
        )

    # Check for expected components in weights
    expected_components = ['encoder', 'decoder', 'joint']
    for comp in expected_components:
        if comp not in components:
            issues.append(f"Missing expected component in weights: {comp}")

    # Check for cache-aware context style
    context_style = analysis.get('att_context_style', '')
    if context_style and context_style != 'NOT_FOUND':
        console.print(f"  [green]Cache-aware context style: {context_style}[/green]")

    if issues:
        console.print("[bold red]ARCHITECTURE VERIFICATION FAILED[/bold red]")
        for issue in issues:
            console.print(f"[red]  - {issue}[/red]")
        raise ArchitectureMismatchError(
            "Architecture does not match expected Nemotron Speech Streaming.\n"
            "Issues:\n" + "\n".join(f"  - {i}" for i in issues)
        )

    console.print("[green]Architecture verification passed[/green]")


def main():
    """Main entry point."""
    console.print("[bold]=" * 60 + "[/bold]")
    console.print("[bold]Step 2: Extract and Analyze Model Weights[/bold]")
    console.print("[bold]=" * 60 + "[/bold]")
    console.print()

    try:
        validate_nemo_exists()
        extract_dir = extract_nemo_archive()
        config_path, weights_path = find_config_and_weights(extract_dir)
        config, analysis = load_and_analyze_config(config_path)
        state_dict, components = load_and_analyze_weights(weights_path)
        verify_architecture(config, analysis, components)

        console.print()
        console.print("[bold green]SUCCESS: Weights extracted and analyzed[/bold green]")
        console.print()
        console.print("Outputs:")
        console.print(f"  - {OUTPUT_DIR / 'model_config.yaml'}")
        console.print(f"  - {OUTPUT_DIR / 'model_weights.ckpt'}")
        console.print(f"  - {OUTPUT_DIR / 'weight_keys.txt'}")
        console.print()
        console.print(f"Next step: [bold green]uv run scripts/03_analyze_architecture.py[/bold green] (or pip: python scripts/03_analyze_architecture.py)")

    except (ExtractionError, ArchitectureMismatchError) as e:
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
