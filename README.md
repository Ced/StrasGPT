## StrasGPT: AI Parallelization Project

<p align="center">
  <img src="assets/llama_math-info.png" width="300" height="300" alt="Cute Llama">
</p>

This project is a direct C implementation of the LLaMa 3.x LLM transformer architecture, reusing the tokenizer and the sampler of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) project and its fork by James Delancey [llama3.c](https://github.com/jameswdelancey/llama3.c) (we warmly thank you!). Given an input prompt, StrasGPT can generate a text that continues it.

## Get and compile StrasGPT

```bash
git clone git@gitlab.unistra.fr:bastoul/strasgpt.git
cd strasgpt
make
```

There are several other building targets:
- `make parallel` to build with MPI and OpenMP support
- `make asan` for Clang's address sanitizer support and debug mode
- `make debug` for debug mode, ideal when using Valgrind

## Get the model files

StrasGPT is directly compatible with HuggingFace format. At the university, files are directly provided on Parallel Programming virtual machines (see `~/partage/` folder). You can also download the smallest models directly from Seafile:
- Password is `mastermathinfo`
- [Llama 3.2 1B (2.5 GB)](https://seafile.unistra.fr/d/64f6233917764f3aad29/)
- [Llama 3.2 3B (6.5 GB)](https://seafile.unistra.fr/d/665bfbc151b149458818/)
- [Llama 3.1 8B (12 GB)](https://seafile.unistra.fr/d/6e5da22479474500acd1/)

Out of the university, we can get LLaMa 3.x checkpoints from HuggingFace. In this case you will need to create an [HuggingFace Account](https://huggingface.co/), and get an access token (click on your profile icon, then "Access Tokens"). Finally you'll need to login then to download the desired models, e.g.:

```bash
pip install 'huggingface_hub[cli]'
huggingface-cli login
git clone https://huggingface.co/meta-llama/Llama-3.2-1B
git clone https://huggingface.co/meta-llama/Llama-3.2-3B
git clone https://huggingface.co/meta-llama/Llama-3.1-8B
git clone https://huggingface.co/meta-llama/Llama-3.1-70B
```

## Run StrasGPT

Run StrasGPT with `-h` option to get all possible options. Here is an example of a command line with a 8-token long prompt and asking to generate 16 tokens (beyond the one generated from prompt analysis):

```bash
./strasgpt -m ../model_zoo/Llama-3.2-1B/ -p "Once upon a time there were three" -n 17
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

Prompt processing (prefill):  8 tokens in 3.802 s (2.104156 token/s)
Token generation  (decode):  16 tokens in 9.333 s (1.714347 token/s)
```

Pretty slow, but you'll soon improve that!