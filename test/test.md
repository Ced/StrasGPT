## Regression tests

The regression tests need Python, NumPy, PyTorch, safetensors and the
Hugging Face tokenizers package. Install these in your Python environment,
then run:

```bash
python3 -m pip install numpy torch safetensors tokenizers
make test
```

Use `make test PYTHON=/path/to/python` to select another Python environment.
The test generates temporary synthetic checkpoints; no download is needed.
Test binaries are separate from the normal executable and build objects.

`make test-parallel` additionally compares sequential and 10-thread execution
(requires the same MPI/OpenMP tools as `make parallel`). `make test-asan`
runs the regression with AddressSanitizer. All targets accept `PYTHON=...`.

The test checks all MXFP4 codes/scales, including signed zero and NaN scales.
Dot-product checks cover every code at every block position and scale, plus
mixed scales, cancellation, odd block counts, long rows and unaligned inputs.
Two tiny layers exercise attention biases, sinks, sliding windows and MoE
routing, expert biases, clipping and weighted expert summation. It compares
the last token's FFN intermediates at each layer and every token's final
logits against an independent PyTorch FP32 reference. Cases include one or
multiple selected experts, absent expert biases, disabled clipping, cached
decoding and a sequence crossing the internal 512-token chunk boundary.

Decoded weights and routing indices are checked exactly. Reference arithmetic
uses `atol=3e-4, rtol=2e-4` for these small fixtures because FP32 reductions
can round differently. Chunk sizes and thread counts must produce identical
binary outputs. Failures report the first mismatching intermediate and index.
These tests cover the MoE path with ordinary RoPE, not YaRN or native BF16
reference execution.

### Configuration tests

After building, `python3 test/config/config.py ./strasgpt` checks scalar and
array EOS token IDs, including malformed arrays, using temporary files.
It needs only Python's standard library and also works with `make asan`.

### Tokenizer tests

`make test-tokenizer` runs small synthetic comparisons with Hugging Face without
model downloads; `make test-tokenizer-asan` uses AddressSanitizer. These checks
are also included in `make test`, `make test-parallel` and `make test-asan`.
Python dependencies are needed only for testing.
All targets accept `PYTHON=...`.

The tests cover all five supported splitting patterns, shuffled token IDs and
vocabulary order, both merge-list formats, JSON escapes, special tokens and all
256 byte values on output. French cases include precomposed and decomposed
common accents, ligatures, curly apostrophes and ordinary and narrow nonbreaking
spaces.

Metaspace BPE is also supported for older Mistral models such as Mistral-7B:
the U+2581 space marker, `prepend_scheme=first`, `split=false`, and byte fallback.
TinyLlama's `Prepend` + `Replace` normalizer with no pre-tokenizer is also
supported. It prefixes every text segment, including after special tokens.
Other Metaspace variants are rejected.
Vocabulary lookup retains raw spellings separately from decoded bytes.
Complete sequences strip one leading space as specified by the decoder;
incremental token output preserves bytes, including partial UTF-8 sequences.
The tests compare token IDs and decoded sequences with Hugging Face and
check all 256 fallback bytes, added-token boundaries, and rejected settings.

To additionally compare a local checkpoint's tokenizer:

```bash
make test-tokenizer TOKENIZER_MODELS=../model_zoo/Qwen3.5-0.8B
```
