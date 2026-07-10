#!/usr/bin/env python3
import sys
import argparse
import re
from transformers import AutoTokenizer

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().rstrip()

def parse_tokens(text):
    parts = re.split(r"[\s,]+", text.strip())
    if not parts or parts == [""]:
        raise ValueError("no token IDs found")
    return [int(p) for p in parts if p]

def write_text(out_fp, text):
    out_fp.write(text)
    out_fp.flush()

def main():
    argp = argparse.ArgumentParser()

    argp.add_argument("--model", "-m",
                      default="meta-llama/Meta-Llama-3-8B",
                      help="Tokenizer to use (default: meta-llama/Meta-Llama-3-8B)")

    argp.add_argument("--tokens", "-t", type=str,
                      help="Token IDs string to decode")

    argp.add_argument("--file", "-f", type=str,
                      help="File containing token IDs to decode")

    argp.add_argument("--output", "-o", type=str,
                      help="Write decoded text to this file (default: stdout)")

    argp.add_argument("--skip_special", action="store_true",
                      help="Skip special tokens when decoding")

    args = argp.parse_args()

    if (args.tokens is None) == (args.file is None):
        print("error: provide exactly one of --tokens/-t or --file/-f",
              file=sys.stderr)
        sys.exit(2)

    raw = args.tokens if args.tokens is not None else read_file(args.file)
    token_ids = parse_tokens(raw)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    text = tokenizer.decode(token_ids, skip_special_tokens=args.skip_special)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_fp:
            write_text(out_fp, text)
    else:
        write_text(sys.stdout, text)

if __name__ == "__main__":
    main()
