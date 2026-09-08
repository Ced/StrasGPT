"""Compare the C tokenizer with Hugging Face, without downloading models."""

import argparse
import copy
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unicodedata

from tokenizers import AddedToken, Regex, Tokenizer, models, normalizers
from tokenizers import decoders, pre_tokenizers, trainers

CONTRACTIONS = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
SPACE = r"|\s*[\r\n]+|\s+(?!\S)|\s+"
WORD = r"[^\r\n\p{L}\p{N}]?\p{L}+"
MARK_WORD = r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
CASE_WORDS = [
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
]
PUNCTUATION = r" ?[^\s\p{L}\p{N}]+[\r\n]*"
PATTERNS = [
    CONTRACTIONS + "|" + WORD + r"|\p{N}|" + PUNCTUATION + SPACE,
    CONTRACTIONS + "|" + MARK_WORD + r"|\p{N}|"
    + r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*" + SPACE,
    CONTRACTIONS + "|" + WORD + r"|\p{N}{1,3}|" + PUNCTUATION + SPACE,
    "|".join(CASE_WORDS) + r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*" + SPACE,
    "|".join(x + CONTRACTIONS + "?" for x in CASE_WORDS)
    + r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*" + SPACE,
]


def prompts():
    texts = [
        "", "Once upon a time", "HelloWORLD helloWorld XMLParser",
        "\u00c9l\u00e8ve, d\u00e9j\u00e0, No\u00ebl, o\u00f9, c\u0153ur",
        "L'\u00e9t\u00e9, l\u2019hiver : \u00ab fran\u00e7ais \u00bb !",
        "\u0152UVRE \u0178 \u00ff \u00c7a va ?", "e\u0301 E\u0301 c\u0327",
        "\u0301e \u0301", "x\u00a0y\u202fz", "1234567890 12,345.67",
        "I'm I'M we've WE'VE it's IT'S 're", '"quote" \\ slash /',
        "a\r\nb\n\n  c\t\td  \n", "  a   b    ", "\v\f\r\t ",
        "\U0001f600\U0001f60a", "<|test|>bonjour<|test_long|>",
        "abcabcabc", "ab bc abc", "word " * 600, "<|\U0001f600|>",
        "\u00aa\u00ba\u00b5 \u00b2\u00b3\u00b9\u00bc\u00bd\u00be",
    ]
    rng = random.Random(42)
    alphabet = ("aAbBcCeE0123 '\"!?,./\\\r\n\t"
                "\u00e9\u00e8\u00c9\u00e0\u00e7\u0153\u0178"
                "\u00a0\u202f\u2019")
    texts += ["".join(rng.choices(alphabet, k=rng.randrange(1, 60)))
              for _ in range(250)]
    return texts


def compare(binary, path, texts):
    reference = Tokenizer.from_file(str(path / "tokenizer.json"))
    data = json.loads((path / "tokenizer.json").read_text())
    result = subprocess.run(
        [str(binary), str(path)],
        input="".join(x.encode().hex() + "\n" for x in texts),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 2 * len(texts), result.stdout
    for i, text in enumerate(texts):
        expected = reference.encode(text, add_special_tokens=False).ids
        actual = list(map(int, lines[2 * i].split()))
        assert actual == expected, (str(path), repr(text), actual, expected)
        decoded = bytes.fromhex(lines[2 * i + 1])
        if data["model"].get("byte_fallback"):
            normalized = reference.decode(expected, skip_special_tokens=False)
        else:
            normalized = (unicodedata.normalize("NFC", text)
                          if data.get("normalizer") else text)
        assert decoded == normalized.encode(), (repr(text), decoded)
    print(f"{path.name}: {len(texts)} reference comparisons passed")


def fixture(path, index, texts):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERNS[index]), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    if index < 2:
        tokenizer.normalizer = normalizers.NFC()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    ))
    tokenizer.add_special_tokens([
        AddedToken("<|test|>", normalized=False, special=True),
        AddedToken("<|test_long|>", normalized=False, special=True),
        AddedToken("<|\U0001f600|>", normalized=False, special=True),
    ])
    data = json.loads(tokenizer.to_str())
    data["model"]["ignore_merges"] = index >= 2
    # Neither JSON object order nor token IDs determine pair ranks.
    vocab = data["model"]["vocab"]
    count = len(vocab)
    data["model"]["vocab"] = {k: count - 1 - v
                              for k, v in reversed(list(vocab.items()))}
    if index % 2:
        data["model"]["merges"] = [" ".join(x)
                                   for x in data["model"]["merges"]]
    path.mkdir()
    # Exercise both Unicode escapes (including surrogate pairs) and UTF-8.
    (path / "tokenizer.json").write_text(json.dumps(
        data, ensure_ascii=index % 2 == 0,
    ))
    return data


def metaspace_fixture(path, normalize=False):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="\u2581", prepend_scheme="first", split=False,
    )
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace("\u2581", " "), decoders.ByteFallback(),
        decoders.Fuse(), decoders.Strip(" ", 1, 0),
    ])
    tokenizer.train_from_iterator([
        "Hello world", "Once upon a time", "a ab abc", "hello  world ",
    ], trainers.BpeTrainer(vocab_size=80, special_tokens=["<unk>"],
                           initial_alphabet=["\u2581"], show_progress=False))
    tokenizer.add_special_tokens([
        AddedToken("<|test|>", normalized=False, special=True),
        AddedToken("<|test_long|>", normalized=False, special=True),
    ])
    tokenizer.add_special_tokens([
        AddedToken("|", normalized=False, special=True),
    ])
    if normalize:
        tokenizer.pre_tokenizer = None
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.Prepend("\u2581"),
            normalizers.Replace(" ", "\u2581"),
        ])
    data = json.loads(tokenizer.to_str())
    vocab = data["model"]["vocab"]
    for token in data["added_tokens"]:
        vocab[token["content"]] = token["id"]
    first = max(list(vocab.values()) +
                [t["id"] for t in data["added_tokens"]]) + 1
    for i in range(256):
        vocab[f"<0x{i:02X}>"] = first + i
    count = len(vocab)
    data["model"]["vocab"] = {k: count - 1 - v
                              for k, v in reversed(list(vocab.items()))}
    for token in data["added_tokens"]:
        token["id"] = count - 1 - token["id"]
    data["model"]["merges"] = [" ".join(pair)
                               for pair in data["model"]["merges"]]
    path.mkdir()
    (path / "tokenizer.json").write_text(json.dumps(data))
    return data


def invalid_metaspace(binary, path, data):
    cases = []
    for key, value in [("split", True), ("replacement", "_"),
                       ("prepend_scheme", "always")]:
        wrong = copy.deepcopy(data)
        wrong["pre_tokenizer"][key] = value
        cases.append(wrong)
    wrong = copy.deepcopy(data)
    wrong["pre_tokenizer"] = {"type": "Sequence", "pretokenizers": [
        wrong["pre_tokenizer"], copy.deepcopy(wrong["pre_tokenizer"]),
    ]}
    cases.append(wrong)
    wrong = copy.deepcopy(data)
    wrong["model"]["byte_fallback"] = False
    cases.append(wrong)
    wrong = copy.deepcopy(data)
    del wrong["model"]["vocab"]["<0xFF>"]
    cases.append(wrong)
    wrong = copy.deepcopy(data)
    wrong["decoder"]["decoders"][-1]["start"] = 0
    cases.append(wrong)
    wrong = copy.deepcopy(data)
    wrong["decoder"]["decoders"].reverse()
    cases.append(wrong)
    for wrong in cases:
        (path / "tokenizer.json").write_text(json.dumps(wrong))
        result = subprocess.run([str(binary), str(path)],
                                capture_output=True, text=True)
        assert result.returncode != 0, wrong
        assert "AddressSanitizer" not in result.stderr, result.stderr
    print(f"{len(cases)} unsupported Metaspace configurations rejected")


def invalid_normalizer(binary, path, data):
    cases = []
    for index, key, value in [(0, "prepend", "_"),
                              (1, "pattern", {"String": "_"}),
                              (1, "content", "_")]:
        wrong = copy.deepcopy(data)
        wrong["normalizer"]["normalizers"][index][key] = value
        cases.append(wrong)
    wrong = copy.deepcopy(data)
    wrong["normalizer"]["normalizers"].reverse()
    cases.append(wrong)
    wrong = copy.deepcopy(data)
    wrong["normalizer"]["normalizers"].pop()
    cases.append(wrong)
    for wrong in cases:
        (path / "tokenizer.json").write_text(json.dumps(wrong))
        result = subprocess.run([str(binary), str(path)],
                                capture_output=True, text=True)
        assert result.returncode != 0, wrong
        assert "AddressSanitizer" not in result.stderr, result.stderr
    print(f"{len(cases)} unsupported normalizer configurations rejected")


def invalid(binary, path, data):
    cases = []
    wrong = copy.deepcopy(data)
    wrong["model"]["vocab"]["bad"] = -1
    cases.append(json.dumps(wrong))
    wrong = copy.deepcopy(data)
    wrong["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "unknown"
    cases.append(json.dumps(wrong))
    wrong = copy.deepcopy(data)
    wrong["added_tokens"][0]["lstrip"] = True
    cases.append(json.dumps(wrong))
    cases += [r'{"bad":"\uD800"}', r'{"bad":"\uDC00"}',
              r'{"bad":"\u0000"}', r'{"bad":"\q"}']
    for text in cases:
        (path / "tokenizer.json").write_text(text)
        result = subprocess.run([str(binary), str(path)],
                                capture_output=True, text=True)
        assert result.returncode != 0, text
        assert "AddressSanitizer" not in result.stderr, result.stderr
    print(f"{len(cases)} malformed/unsupported inputs rejected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("models", nargs="*", type=Path)
    args = parser.parse_args()
    texts = prompts()
    with tempfile.TemporaryDirectory(prefix="strasgpt-tokenizer-") as tmp:
        for i in range(5):
            path = Path(tmp) / f"pattern-{i}"
            data = fixture(path, i, texts)
            compare(args.binary.resolve(), path, texts)
        invalid(args.binary.resolve(), path, data)
        path = Path(tmp) / "metaspace"
        data = metaspace_fixture(path)
        compare(args.binary.resolve(), path, texts + [
            " hello", "  hello", "\u2581hello", "<|test|>hello",
            "<|test|> hello", "a<|test|>b", "\u4e2d\u6587\U0010ffff",
        ])
        invalid_metaspace(args.binary.resolve(), path, data)
        path = Path(tmp) / "prepend-replace"
        data = metaspace_fixture(path, normalize=True)
        compare(args.binary.resolve(), path, texts + [
            " hello", "  hello", "\u2581hello", "<|test|>hello",
            "<|test|> hello", "a<|test|>b", "|x" * 200, "x|" * 200,
            "<|test|><|test_long|>", "\u4e2d\u6587\U0010ffff",
        ])
        invalid_normalizer(args.binary.resolve(), path, data)
    for path in args.models:
        compare(args.binary.resolve(), path, texts)


if __name__ == "__main__":
    main()
