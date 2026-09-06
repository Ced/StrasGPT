## StrasGPT: LLMs Are Easier Than You Think

<p align="center">
  <img src="assets/llama_math-info.png" width="300" height="300" alt="Cute Llama">
</p>

This program is a direct C implementation of the Qwen 3.5 / Qwen3 / LLaMa 3.x / Mistral LLM transformer architecture, reusing the tokenizer and the sampler of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) project and its fork by James Delancey [llama3.c](https://github.com/jameswdelancey/llama3.c) (we warmly thank you!). Given an input prompt, StrasGPT can generate a text that continues it, and it also supports basic instruct mode. It was initially designed as a parallel programming project for master students in 2025 (students had to parallelize it with OpenMP + MPI). It is now getting continued for fun and (polyhedral) compiler research.

## Get and compile StrasGPT

First, you need git, basic compiler tools and (optionally) parallel computing libraries to get and compile the tool. E.g., from a fresh Ubuntu distribution (3rd line is optional but recommended to compile the parallel, faster version):

```bash
sudo apt update
sudo apt install git make clang bison flex
sudo apt install libopenmpi-dev openmpi-bin openmpi-common
```

Then let's get and compile it:

```bash
git clone git@github.com:Ced/StrasGPT.git
cd strasgpt
make
```

There are several other building targets:
- `make parallel` to build the faster parallel version, that target requires mpicc compiler
- `make asan` for Clang's address sanitizer support and debug mode, that target requires Clang compiler
- `make debug` for debug mode, ideal when using Valgrind

## Get the model files

You can use, e.g., Qwen 3.5, GPT-OSS, LLaMa 3.x or Mistral checkpoints from HuggingFace. You will need to create an [HuggingFace Account](https://huggingface.co/), and get an access token (click on your profile icon, then "Access Tokens"). Finally you'll need to login then to download the desired models, e.g. here are some tested models:

```bash
pip install 'huggingface_hub[cli]'
huggingface-cli login
git clone https://huggingface.co/openai/gpt-oss-20b
git clone https://huggingface.co/Qwen/Qwen3.5-0.8B
git clone https://huggingface.co/Qwen/Qwen3-0.6B
git clone https://huggingface.co/Qwen/Qwen3-4B
git clone https://huggingface.co/Qwen/Qwen3-14B
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
git clone https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B
git clone https://huggingface.co/meta-llama/Llama-3.2-1B
git clone https://huggingface.co/meta-llama/Llama-3.2-3B
git clone https://huggingface.co/meta-llama/Llama-3.1-8B
git clone https://huggingface.co/mistralai/Mistral-Nemo-Base-2407
git clone https://huggingface.co/mistralai/Ministral-8B-Instruct-2410
git clone https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501
```

## Run StrasGPT

Run StrasGPT with `-h` option to get all possible options. Here is an example of a command line with a 8-token long prompt and asking to generate 16 tokens (beyond the one generated from prompt analysis) and using 10 threads:

```bash
./strasgpt -m ../model_zoo/Llama-3.2-1B/ -p "Once upon a time there were three" -n 17 -t 10
```

And here is the output on my M4 Mac:

```
...
Transformer:
- Configuration:
--- embedding_dim:      2048
--- hidden_dim:         8192
--- layer_count:        16
--- q_head_count:       32
--- kv_head_count:      8
--- vocabulary_len:     128256
--- context_len:        131072
--- aliased_out_weight: true
...

[Once upon a time there were three] little pigs.
Three little pigs went out for a pig walk. They heard music playing

Max memory used (RSS): 2.37 GB
Prompt processing (prefill):    8 tokens in   0.057 s (140.350877 token/s)
Token generation  (decode):    16 tokens in   0.213 s (79.207921 token/s)
```

Actually not that bad!
## Regression tests

The small GPT-OSS/MoE regression test needs Python, NumPy, PyTorch and
safetensors. Install these in your Python environment, then run:

```bash
python3 -m pip install numpy torch safetensors
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
