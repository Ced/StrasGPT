#!/usr/bin/env python3
import sys
import argparse
from transformers import AutoTokenizer

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().rstrip()

def write_tokens(out_fp, token_list):
    out_fp.write(", ".join(str(t) for t in token_list))
    out_fp.flush()

def main():
    argp = argparse.ArgumentParser()

    argp.add_argument("--model", "-m",
                      default="meta-llama/Meta-Llama-3-8B",
                      help="Tokenizer to use (default: meta-llama/Meta-Llama-3-8B)")

    argp.add_argument("--prompt", "-p", type=str,
                      help="Prompt string to encode")

    argp.add_argument("--file", "-f", type=str,
                      help="Text file to encode")

    argp.add_argument("--output", "-o", type=str,
                      help="Write tokens to this file (default: stdout)")

    argp.add_argument("--no_bos", action="store_true",
                      help="Do not prepend the model BOS token")

    args = argp.parse_args()

    if (args.prompt is None) == (args.file is None):
        print("error: provide exactly one of --prompt/-p or --file/-f",
              file=sys.stderr)
        sys.exit(2)

    text = args.prompt if args.prompt is not None else read_file(args.file)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokens = []

    # Prepend BOS token if defined and not disabled
    if not args.no_bos:
        if tokenizer.bos_token_id is not None:
            tokens.append(tokenizer.bos_token_id)

    # Encode text WITHOUT auto special tokens
    tokens.extend(tokenizer.encode(text, add_special_tokens=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_fp:
            write_tokens(out_fp, tokens)
    else:
        write_tokens(sys.stdout, tokens)

if __name__ == "__main__":
    main()
