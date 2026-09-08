## StrasGPT: LLMs Are Easier Than You Think

<p align="center">
  <img src="assets/llama_math-info.png" width="300" height="300" alt="Cute Llama">
</p>

This program is a direct C implementation of the transformer architecture, capable
to run inference for, e.g., Qwen 3.5 / GPT-OSS / Mistral / LLaMa 3.x LLMs.
It originally derived from Andrej Karpathy's
[llama2.c](https://github.com/karpathy/llama2.c) project and its fork by James Delancey
[llama3.c](https://github.com/jameswdelancey/llama3.c) (we warmly thank you!).
Given an input prompt, StrasGPT can generate a text that continues it, and it
also supports basic interactive chat. It was initially designed as a parallel
programming project for master students in 2025 (students had to parallelize it
with OpenMP + MPI). It is now getting continued for fun and (polyhedral)
compiler research.

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

You can use, e.g., Qwen (2.5, 3, 3.5), GPT-OSS, LLaMa 3.x or Mistral (Ministral, Nemo, Small) checkpoints from HuggingFace.
Some open models are provided through direct links for convenience:
- [GPT-OSS 20b (13.8 GB)](https://seafile.unistra.fr/f/981ea64f2f204db6a444/?dl=1)
- [Qwen 3.5 0.8B (1.8 GB)](https://seafile.unistra.fr/f/11eb47d34e6d4edc89ec/?dl=1)
- [Qwen 3.5 4B (9.3 GB)](https://seafile.unistra.fr/f/9c070096fa454283a842/?dl=1)
- [Qwen 3 0.6B (1.5 GB)](https://seafile.unistra.fr/f/6ec48bc9b8d04f0aad30/?dl=1)
- [Qwen 3 4B Instruct (8.1 GB)](https://seafile.unistra.fr/f/6ec48bc9b8d04f0aad30/?dl=1)
- [Ministral 3 3B Instruct (7.7 GB)](https://seafile.unistra.fr/f/af681696bddd472e8c04/?dl=1)
- [Mistral Nemo 12B Instruct 2407 (24.5 GB)](https://seafile.unistra.fr/f/f83cc292583c48318784/?dl=1)
- [Mistral Small 24B Base 2501 (47.2GB)](https://seafile.unistra.fr/f/96f6e03c73594d09a5aa/?dl=1)

To get LLaMa model and others, you will need to create an [HuggingFace Account](https://huggingface.co/), and get an access token (click on your profile icon, then "Access Tokens"). Finally you'll need to login then to download the desired models, e.g. here are some tested models:

```bash
pip install --upgrade huggingface_hub
hf auth login
hf download openai/gpt-oss-20b --local-dir ./gpt-oss-20b
hf download Qwen/Qwen3.5-0.8B --local-dir ./Qwen3.5-0.8B
hf download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B
hf download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B
hf download Qwen/Qwen3-4B --local-dir ./Qwen3-4B
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir ./Qwen3-4B-Instruct-2507
hf download Qwen/Qwen3-14B --local-dir ./Qwen3-14B
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir ./Qwen3-VL-2B-Instruct
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir ./Qwen3-VL-4B-Instruct
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ./Qwen3-VL-8B-Instruct
hf download Qwen/Qwen2.5-0.5B --local-dir ./Qwen2.5-0.5B
hf download meta-llama/Llama-3.2-1B --local-dir ./Llama-3.2-1B
hf download meta-llama/Llama-3.2-3B --local-dir ./Llama-3.2-3B
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir ./Llama-3.2-3B-Instruct
hf download meta-llama/Llama-3.1-8B --local-dir ./Llama-3.1-8B
hf download mistralai/Ministral-3-3B-Instruct-2512-BF16 --local-dir ./Ministral-3-3B-Instruct-2512-BF16
hf download mistralai/Ministral-8B-Instruct-2410 --local-dir ./Ministral-8B-Instruct-2410
hf download mistralai/Mistral-Nemo-Base-2407 --local-dir ./Mistral-Nemo-Base-2407
hf download mistralai/Mistral-Nemo-Instruct-2407 --local-dir ./Mistral-Nemo-Instruct-2407
hf download mistralai/Mistral-Small-24B-Base-2501 --local-dir ./Mistral-Small-24B-Base-2501
hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ./Mistral-7B-Instruct-v0.3
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./TinyLlama-1.1B-Chat-v1.0
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

## Chat

Use `--chat` with a Qwen, Mistral/Ministral, Llama 3 instruct or gpt-oss
model:

```bash
./strasgpt --chat -m ../model_zoo/Qwen3.5-0.8B -n 256 -s 42
./strasgpt --chat -m ../model_zoo/Llama-3.2-3B-Instruct -n 256 -t 10
./strasgpt --chat -m ../model_zoo/Ministral-3-3B-Instruct-2512-BF16 -n 256 -t 10
```

Enter one message per line. Empty lines are ignored; `/quit` or EOF exits.
Conversation history stays in the model cache. `-n` limits each reply; a
truncated reply is closed before the next user message. When the context is
full, start a new session. Qwen 3.5 uses its non-thinking prefix and gpt-oss
uses the final channel for direct answers.

Chat selects the message format automatically.
It reads stdin and cannot be combined with `-p`, `-f` or `--pre-tokenized`.
Replies go to stdout and interface labels go to stderr.

## Tests

See `./test/test.md`
