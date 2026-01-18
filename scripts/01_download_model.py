#!/usr/bin/env python3
"""
01_download_model.py - Download Nemotron Speech Streaming model from HuggingFace

This script downloads the official .nemo file from:
https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b

WHY THIS APPROACH:
- We download the original .nemo file directly from NVIDIA's official repo
- We do NOT use any third-party conversions (which may be mislabeled)
- We verify the downloaded file matches expected properties

ERROR HANDLING:
- Script fails loudly if download fails
- Script fails if file doesn't match expected properties
- No fallbacks or "try again" logic - failure means investigation needed
"""

import os
import sys
from pathlib import Path

# Fail early if imports don't work
try:
    from huggingface_hub import hf_hub_download, HfApi
    from rich.console import Console
    from rich.progress import Progress
except ImportError as e:
    print(f"FATAL: Missing required package: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# ============================================================
# STRICT CONFIGURATION - DO NOT MODIFY WITHOUT DOCUMENTATION
# ============================================================

# The ONLY correct model ID - any other is a different model
MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"

# The exact filename we expect
EXPECTED_FILENAME = "nemotron-speech-streaming-en-0.6b.nemo"

# Expected file size in bytes (approximately 2.47 GB based on HF listing)
# We use a range because exact size may vary slightly
EXPECTED_SIZE_MIN_BYTES = 2_400_000_000  # 2.4 GB minimum
EXPECTED_SIZE_MAX_BYTES = 2_600_000_000  # 2.6 GB maximum

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"

console = Console()


class DownloadValidationError(Exception):
    """Raised when downloaded file doesn't match expectations."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def validate_environment():
    """Validate that the environment is correctly configured."""
    console.print("[bold blue]Validating environment...[/bold blue]")

    # Check output directory exists or can be created
    if not OUTPUT_DIR.exists():
        console.print(f"Creating output directory: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_DIR.is_dir():
        raise ConfigurationError(f"Output path exists but is not a directory: {OUTPUT_DIR}")

    console.print("[green]Environment validation passed[/green]")


def verify_model_exists_on_hub():
    """Verify the model exists on HuggingFace Hub before downloading."""
    console.print(f"[bold blue]Verifying model exists: {MODEL_ID}[/bold blue]")

    api = HfApi()

    try:
        model_info = api.model_info(MODEL_ID)
    except Exception as e:
        raise ConfigurationError(
            f"Cannot access model '{MODEL_ID}' on HuggingFace Hub.\n"
            f"This could mean:\n"
            f"  1. The model ID is wrong\n"
            f"  2. The model requires authentication\n"
            f"  3. Network connectivity issues\n"
            f"Original error: {e}"
        )

    # Verify the .nemo file exists in the repo
    files = [f.rfilename for f in model_info.siblings]

    if EXPECTED_FILENAME not in files:
        raise ConfigurationError(
            f"Expected file '{EXPECTED_FILENAME}' not found in model repo.\n"
            f"Available files: {files}\n"
            f"This indicates the model structure has changed - investigation required."
        )

    console.print(f"[green]Model verified: {model_info.id}[/green]")
    console.print(f"  - Last modified: {model_info.last_modified}")
    console.print(f"  - Files: {len(files)}")

    return model_info


def download_model():
    """Download the .nemo file from HuggingFace."""
    console.print(f"[bold blue]Downloading {EXPECTED_FILENAME}...[/bold blue]")
    console.print("[yellow]This is a 2.5GB file - may take several minutes[/yellow]")

    try:
        downloaded_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=EXPECTED_FILENAME,
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False,  # We want actual files, not symlinks
        )
    except Exception as e:
        raise DownloadValidationError(
            f"Download failed.\n"
            f"Original error: {e}\n"
            f"Do NOT retry automatically - investigate the failure first."
        )

    downloaded_path = Path(downloaded_path)
    console.print(f"[green]Downloaded to: {downloaded_path}[/green]")

    return downloaded_path


def validate_downloaded_file(file_path: Path):
    """Validate the downloaded file matches our expectations."""
    console.print("[bold blue]Validating downloaded file...[/bold blue]")

    # Check file exists
    if not file_path.exists():
        raise DownloadValidationError(f"Downloaded file does not exist: {file_path}")

    # Check file size
    file_size = file_path.stat().st_size
    console.print(f"  File size: {file_size:,} bytes ({file_size / 1e9:.2f} GB)")

    if file_size < EXPECTED_SIZE_MIN_BYTES:
        raise DownloadValidationError(
            f"File too small: {file_size:,} bytes\n"
            f"Expected at least: {EXPECTED_SIZE_MIN_BYTES:,} bytes\n"
            f"This may indicate a truncated download or wrong file."
        )

    if file_size > EXPECTED_SIZE_MAX_BYTES:
        raise DownloadValidationError(
            f"File too large: {file_size:,} bytes\n"
            f"Expected at most: {EXPECTED_SIZE_MAX_BYTES:,} bytes\n"
            f"This may indicate the model has changed significantly."
        )

    # Check file is readable and has expected structure (.nemo is a tarball)
    import tarfile

    if not tarfile.is_tarfile(file_path):
        raise DownloadValidationError(
            f"File is not a valid tar archive: {file_path}\n"
            f".nemo files are tar archives containing model_config.yaml and model_weights.ckpt"
        )

    # Peek inside to verify expected structure
    with tarfile.open(file_path, 'r') as tar:
        members = tar.getnames()

    expected_members = ['model_config.yaml', 'model_weights.ckpt']
    missing = [m for m in expected_members if m not in members and f"./{m}" not in members]

    if missing:
        raise DownloadValidationError(
            f"Archive missing expected files: {missing}\n"
            f"Archive contains: {members[:10]}{'...' if len(members) > 10 else ''}\n"
            f"This indicates an unexpected .nemo file structure."
        )

    console.print("[green]File validation passed[/green]")
    console.print(f"  Archive contains: {len(members)} files")

    return True


def main():
    """Main entry point."""
    console.print("[bold]=" * 60 + "[/bold]")
    console.print("[bold]Step 1: Download Nemotron Speech Streaming Model[/bold]")
    console.print("[bold]=" * 60 + "[/bold]")
    console.print()
    console.print(f"Source: {MODEL_ID}")
    console.print(f"File: {EXPECTED_FILENAME}")
    console.print(f"Output: {OUTPUT_DIR}")
    console.print()

    try:
        validate_environment()
        verify_model_exists_on_hub()
        downloaded_path = download_model()
        validate_downloaded_file(downloaded_path)

        console.print()
        console.print("[bold green]SUCCESS: Model downloaded and validated[/bold green]")
        console.print(f"Next step: python scripts/02_extract_weights.py")

    except (ConfigurationError, DownloadValidationError) as e:
        console.print()
        console.print(f"[bold red]FATAL ERROR[/bold red]")
        console.print(f"[red]{e}[/red]")
        console.print()
        console.print("[yellow]This script does not retry or fallback.[/yellow]")
        console.print("[yellow]Investigate and fix the issue before re-running.[/yellow]")
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
