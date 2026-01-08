## StrasGPT: LLMs Are Easier Than You Think

<p align="center">
  <img src="assets/llama_math-info.png" width="300" height="300" alt="Cute Llama">
</p>

This program is a direct C implementation of the LLaMa 3.x LLM transformer architecture, reusing the tokenizer and the sampler of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) project and its fork by James Delancey [llama3.c](https://github.com/jameswdelancey/llama3.c) (we warmly thank you!). Given an input prompt, StrasGPT can generate a text that continues it. It was initially designed as a parallel programming project for master students in 2025 (students had to parallelize it with OpenMP + MPI). It is now getting continued for fun and (polyhedral) compiler research.

## Get and compile StrasGPT

You just need git, a C compiler and make.

```bash
git clone git@gitlab.unistra.fr:bastoul/strasgpt.git
cd strasgpt
make
```

There are several other building targets:
- `make parallel` to build the faster parallel version, that target requires mpicc compiler
- `make asan` for Clang's address sanitizer support and debug mode, that target requires Clang compiler
- `make debug` for debug mode, ideal when using Valgrind

## Get the model files

You can use, e.g., LLaMa 3.x or Mistral checkpoints from HuggingFace. You will need to create an [HuggingFace Account](https://huggingface.co/), and get an access token (click on your profile icon, then "Access Tokens"). Finally you'll need to login then to download the desired models, e.g.:

```bash
pip install 'huggingface_hub[cli]'
huggingface-cli login
git clone https://huggingface.co/meta-llama/Llama-3.2-1B
git clone https://huggingface.co/meta-llama/Llama-3.2-3B
git clone https://huggingface.co/meta-llama/Llama-3.1-8B
git clone https://huggingface.co/mistralai/Mistral-Nemo-Base-2407
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