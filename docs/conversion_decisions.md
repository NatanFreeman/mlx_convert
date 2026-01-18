# Conversion Decisions and Rationale

This document explains every decision made during the MLX conversion process and why each choice was necessary.

## Table of Contents
1. [Model Selection](#model-selection)
2. [Why Not Use Existing Conversions](#why-not-use-existing-conversions)
3. [Weight Transformation Logic](#weight-transformation-logic)
4. [Cache-Aware Streaming](#cache-aware-streaming)
5. [Configuration Mapping](#configuration-mapping)
6. [Known Limitations](#known-limitations)

---

## Model Selection

### Decision: Use nvidia/nemotron-speech-streaming-en-0.6b directly

**Why this specific model?**
- First unified model in Nemotron Speech family
- Supports both streaming and batch workloads
- Cache-aware architecture for efficient real-time inference
- Native punctuation and capitalization (no separate model needed)
- State-of-the-art accuracy (7.16% average WER)

**Why not alternatives?**

| Model | Why Not |
|-------|---------|
| Parakeet-TDT-0.6b-v3 | Uses TDT decoder (different architecture) |
| Parakeet-CTC-1.1b | Uses CTC decoder (no streaming support) |
| Whisper | Different architecture, higher latency |

---

## Why Not Use Existing Conversions

### Decision: Convert from original .nemo, not use animaslabs/nemotron-speech-streaming-en-0.6b-mlx

**The Problem:**
The HuggingFace repo `animaslabs/nemotron-speech-streaming-en-0.6b-mlx` is **mislabeled**. Its model card states:
> "This model was converted to MLX format from nvidia/parakeet-tdt-0.6b-v3"

This is **Parakeet-TDT**, NOT Nemotron Speech Streaming. The key differences:

| Attribute | Parakeet-TDT | Nemotron Speech Streaming |
|-----------|--------------|---------------------------|
| Decoder | TDT (Time Duration Transducer) | RNNT (RNN Transducer) |
| Encoder | Standard FastConformer | Cache-Aware FastConformer |
| Streaming | Via chunked inference | Native cache-based |
| Inference | Stateless | Stateful (caches activations) |

**Why this matters:**
- TDT predicts token + duration simultaneously → faster but different output
- RNNT predicts token or blank at each step → standard transducer behavior
- Using TDT weights with RNNT decoder = undefined behavior
- Using RNNT weights with TDT decoder = undefined behavior

**Silent Quality Degradation:**
Using the wrong model won't crash - it will produce transcriptions that seem reasonable but are subtly wrong. This is the worst kind of error.

---

## Weight Transformation Logic

### Decision: Follow parakeet-mlx conversion pattern with modifications

Based on: https://gist.github.com/senstella/77178bb5d6ec67bf8c54705a5f490bed

### 1. Skip Preprocessor Weights

**What:** All weights starting with `preprocessor`

**Why:**
- Preprocessor handles mel-spectrogram extraction
- MLX implementation uses librosa for preprocessing instead
- These weights are ~3-5% of model but not used at inference

**Validation:**
- Verify total parameter count is ~95%+ of original
- Verify all encoder/decoder/joint weights are preserved

### 2. Skip Batch Norm Tracking

**What:** Weights containing `num_batches_tracked`

**Why:**
- These are running statistics for training
- Not used during inference (model should be in eval mode)
- Including them would cause shape mismatches

### 3. Convolution Weight Permutation

**What:**
- 4D tensors: `(out, in, h, w)` → `(out, h, w, in)` via `permute(0, 2, 3, 1)`
- 3D tensors: `(out, in, len)` → `(out, len, in)` via `permute(0, 2, 1)`

**Why:**
- PyTorch uses NCHW (channels first) format
- MLX uses NHWC (channels last) format
- This is a memory layout difference, not a semantic one

**Affected patterns (based on actual Nemotron weight inspection):**
- `encoder.pre_encode.conv.{N}.weight` - 2D convolutions for subsampling (4D tensors)
- `encoder.layers.{N}.conv.*.weight` - 1D convolutions in conformer blocks (3D tensors)

**Excluded (contain 'conv' in key path but are NOT convolutions):**

1. `batch_norm` weights:
   - Example: `encoder.layers.0.conv.batch_norm.weight`
   - Shape: `[1024]` (1D)
   - Reason: BatchNorm scale parameters, not conv kernels

2. `norm_conv` weights:
   - Example: `encoder.layers.0.norm_conv.weight`
   - Shape: `[1024]` (1D)
   - Reason: **This is a LayerNorm, NOT a convolution**, despite the misleading name
   - NeMo naming convention: `norm_X` = "the LayerNorm that precedes module X"
   - Source verification: NeMo's `ConformerLayer` class defines `self.norm_conv = LayerNorm(d_model)`
   - Reference: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/conformer_modules.py

**How this was determined:**
1. Actual weight inspection showed `norm_conv.weight` has shape `[1024]` (1D)
2. Actual conv weights like `conv.depthwise_conv.weight` have shape `[1024, 1, 9]` (3D)
3. NeMo source code confirms `self.norm_conv = LayerNorm(d_model)`

### 4. LSTM Weight Renaming

**What (based on actual Nemotron weight keys):**
- `decoder.prediction.dec_rnn.lstm.weight_ih_l0` → `decoder.prediction.dec_rnn.lstm.0.Wx`
- `decoder.prediction.dec_rnn.lstm.weight_hh_l0` → `decoder.prediction.dec_rnn.lstm.0.Wh`
- `decoder.prediction.dec_rnn.lstm.bias_ih_l0` + `bias_hh_l0` → `decoder.prediction.dec_rnn.lstm.0.bias` (summed)

**Transformation method:**
- Simple string replace (NOT regex)
- `key.replace('weight_ih_l', '') + '.Wx'` → inserts layer number before `.Wx`
- Matches parakeet-mlx conversion gist exactly

**Why:**
- MLX uses different naming conventions for RNN weights
- MLX combines input and hidden biases into single bias tensor
- The sum is mathematically equivalent: `Wx·x + Wh·h + b_ih + b_hh = Wx·x + Wh·h + b`

### 5. Direct Copy (Everything Else)

**What:** All remaining weights are copied with original key names

**Why:**
- Linear layer weights, embeddings, layer norms, etc. are format-compatible
- No transformation needed for these weight types

---

## Cache-Aware Streaming

### Decision: Cache-aware streaming is runtime behavior, not weight-dependent

**Understanding the Architecture:**

The cache-aware mechanism described in arXiv:2312.17279:
1. Encoder processes audio in non-overlapping chunks
2. Self-attention layers cache previous key/value tensors
3. Convolution layers cache previous activations
4. Each frame is processed exactly once (no recomputation)

**Key insight:** The cache mechanism is in the **inference code**, not the weights. The weights are the same whether you use:
- Batch inference (process full audio at once)
- Streaming inference (process chunk by chunk with caches)

**Implication for MLX:**
- Weights convert identically regardless of streaming intent
- Streaming support depends on inference implementation
- parakeet-mlx `transcribe_stream()` should work if it implements caching

**Configuration for streaming:**
```python
# Context size controls attention window
# [left_frames, right_frames]
context_size = [70, 13]  # For 1.12s chunks

# Chunk sizes in 80ms frames:
# [70, 0]  = 80ms   (1 frame, lowest latency)
# [70, 1]  = 160ms  (2 frames)
# [70, 6]  = 560ms  (7 frames)
# [70, 13] = 1120ms (14 frames, best accuracy)
```

---

## Configuration Mapping

### Decision: Use ORIGINAL NeMo config format (converted to JSON)

**CRITICAL FINDING (2026-01-19): parakeet-mlx expects NeMo config format, not a custom schema!**

**How parakeet-mlx loads models (`parakeet_mlx/utils.py`):**
```python
def from_config(config: dict) -> BaseParakeet:
    if (
        config.get("target")  # <-- Expects "target" field, not "_target_"
        == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
        and config.get("model_defaults", {}).get("tdt_durations") is None
    ):
        cfg = from_dict(ParakeetRNNTArgs, config)  # <-- Uses dacite to map dict to dataclass
        model = ParakeetRNNT(cfg)
```

**The config.json must have:**
1. `target`: `"nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"` (note: NOT `_target_`)
2. `preprocessor`: Dict matching `PreprocessArgs` dataclass
3. `encoder`: Dict matching `ConformerArgs` dataclass
4. `decoder`: Dict matching `PredictArgs` dataclass (NOT RNNT config!)
5. `joint`: Dict matching `JointArgs` dataclass
6. `decoding`: Dict matching `RNNTDecodingArgs` dataclass

**What `from_dict()` does (dacite library):**
- Recursively converts nested dicts to dataclass instances
- Extra keys are ignored (good - we can include metadata)
- Missing required keys cause errors (bad - must include all required fields)

**NeMo config structure (from model_config.yaml):**
```yaml
target: nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel
preprocessor:
  sample_rate: 16000
  normalize: NA
  window_size: 0.025
  window_stride: 0.01
  window: hann
  features: 128
  n_fft: 512
  dither: 1.0e-05
  pad_to: 0
  pad_value: 0.0
encoder:
  n_layers: 24
  d_model: 1024
  # ... more fields matching ConformerArgs
decoder:
  blank_as_pad: true
  vocab_size: 1024
  prednet:
    pred_hidden: 640
    pred_rnn_layers: 2
joint:
  num_classes: 1024
  vocabulary: [<unk>, ▁t, ▁th, ...]  # <-- VOCABULARY IS HERE, NOT IN TOKENIZER FILES
  jointnet:
    joint_hidden: 640
    activation: relu
    encoder_hidden: 1024
    pred_hidden: 640
decoding:
  greedy:
    max_symbols: 10
```

**Why the original config.json was WRONG:**
- Used custom schema with `model_type`, `decoder_type`, `source_model` fields
- Missing `target` field entirely
- Missing vocabulary in `joint.vocabulary`
- Field names didn't match parakeet-mlx dataclass expectations

**Solution:**
Convert the NeMo YAML config directly to JSON, keeping the exact same structure.

---

## Known Limitations

### 1. CRITICAL: Causal Downsampling Not Supported

**Status:** BLOCKING for streaming inference

**Issue:** The Nemotron model uses `causal_downsampling: true` in its encoder configuration. parakeet-mlx raises `NotImplementedError` for this case.

**parakeet-mlx conformer.py (lines 355-362):**
```python
if args.subsampling == "dw_striding" and args.causal_downsampling is False:
    self.pre_encode = DwStridingSubsampling(args)
else:
    self.pre_encode = nn.Identity()
    raise NotImplementedError(
        "Other subsampling haven't been implemented yet!"
    )
```

**What causal_downsampling affects:**
- Padding behavior in convolution layers during inference
- Causal mode uses asymmetric padding (all padding on past side) to ensure no future information leaks
- This is a RUNTIME behavior, not a weight structure difference

**Weight compatibility:**
- The weights ARE structurally compatible (same shapes, same key names)
- pre_encode structure: `conv.0`, `conv.2`, `conv.3`, `conv.5`, `conv.6`, `out` (matching parakeet-mlx)
- Only runtime padding differs

**Workaround options:**
1. **Set `causal_downsampling: false` in config** - Model loads but streaming may have audio bleed issues
2. **Modify parakeet-mlx** - Add causal padding support to DwStridingSubsampling
3. **Use batch inference only** - Full audio processing works, just no streaming

**Recommendation:** Set `causal_downsampling: false` and document that streaming inference may have boundary artifacts until parakeet-mlx adds causal support.

### 2. Tokenizer: Vocabulary Embedded in Config, NOT Separate Files

**Status:** CRITICAL FINDING (2026-01-19)

**WRONG assumption:** Tokenizer files (SentencePiece .model/.vocab) are needed for inference.

**CORRECT behavior:** parakeet-mlx gets vocabulary from `config["joint"]["vocabulary"]`:

**parakeet_mlx/tokenizer.py:**
```python
def decode(tokens: list[int], vocabulary: list[str]):
    return "".join([vocabulary[token].replace("▁", " ") for token in tokens])
```

**parakeet_mlx/rnnt.py (JointArgs dataclass):**
```python
@dataclass
class JointArgs:
    num_classes: int
    vocabulary: list[str]  # <-- Vocabulary is a list of 1024 strings
    jointnet: JointNetworkArgs
    num_extra_outputs: int = 0
```

**What this means:**
- The vocabulary (1024 tokens) must be embedded in config.json under `joint.vocabulary`
- This is already present in the NeMo config (lines 90-1114 of model_config.yaml)
- The SentencePiece .model/.vocab files are NOT used by parakeet-mlx
- We can SKIP copying tokenizer files to output (they're unused)

**Why the original approach was wrong:**
- Script 04 copied tokenizer.model and tokenizer.vocab files
- These files are NOT read by parakeet-mlx
- The vocabulary was NOT in config.json
- Result: Model would fail to decode (no vocabulary)

### 3. parakeet-mlx Config Format

**Status:** RESOLVED

**Issue:** parakeet-mlx expects NeMo config format with specific field names, not a custom schema.

**Solution:** Convert NeMo YAML to JSON directly, including:
- `target` field (NOT `_target_`)
- `preprocessor`, `encoder`, `decoder`, `joint`, `decoding` sections
- Vocabulary embedded in `joint.vocabulary`

See "Configuration Mapping" section above for details.

### 3. Beam Search

**Status:** Greedy decoding only

**Issue:** parakeet-mlx notes that beam search is "only available for TDT models for now"

**Mitigation:**
- Greedy decoding works for RNNT
- Beam search would improve accuracy but is not critical

### 4. Quantization

**Status:** Not supported

**Issue:** parakeet-mlx and mlx-audio note they "do not support quantized models"

**Mitigation:**
- Use full precision (fp32/fp16)
- Quantization could be added in future MLX versions

---

## Version Pinning Rationale

All dependencies are pinned to specific versions as of 2026-01-18:

| Package | Version | Why This Version |
|---------|---------|------------------|
| torch | 2.9.1 | Latest stable, needed for .ckpt loading |
| nemo-toolkit | 2.6.1 | Latest stable, matches model requirements |
| safetensors | 0.7.0 | Latest stable, MLX-compatible format |
| huggingface-hub | 1.3.2 | Latest stable, for model download |
| mlx | 0.30.3 | Latest stable (Apple Silicon only) |
| parakeet-mlx | 0.4.1 | Latest stable, RNNT support |

**Why pin versions?**
- Reproducibility: Same inputs → same outputs
- Debugging: Known-good configuration
- Stability: Avoid breaking changes

---

## References

1. Nemotron Speech Model: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
2. Cache-Aware Streaming Paper: https://arxiv.org/abs/2312.17279
3. FastConformer Paper: https://arxiv.org/abs/2305.05084
4. parakeet-mlx: https://github.com/senstella/parakeet-mlx
5. MLX Framework: https://github.com/ml-explore/mlx
6. Conversion Script Reference: https://gist.github.com/senstella/77178bb5d6ec67bf8c54705a5f490bed
