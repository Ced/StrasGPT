"""Small FP32 GPT-OSS reference; no downloaded models are required."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file


class Weights:
    def __init__(self, path):
        self.weights = load_file(str(path / "model.safetensors"))

    def get(self, name, xp=None):
        value = self.weights.get(name)
        if value is None:
            return None
        if xp is not None:
            value = value[xp]
        return value if value.dtype == torch.uint8 else value.float()

    def decode(self, name, xp):
        blocks = self.get(name + "_blocks", xp).long()
        scales = self.get(name + "_scales", xp).int() - 127
        # Decode independently of the C helper, including nibble order.
        codes = torch.stack((blocks & 15, blocks >> 4), -1)
        magnitude = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
        values = magnitude[codes & 7] * (1 - 2 * (codes >> 3))
        return torch.ldexp(values, scales[..., None, None]).flatten(-3)


def fixture(path, expert_count, selected_count, bias, limit):
    torch.manual_seed(42)
    path.mkdir()
    config = dict(
        model_type="gpt_oss", hidden_size=64, head_dim=32,
        intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=520, rms_norm_eps=1e-5,
        rope_theta=10000., sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        num_local_experts=expert_count, num_experts_per_tok=selected_count,
        swiglu_limit=limit,
    )
    (path / "config.json").write_text(json.dumps(config))
    weights = {}

    def random(name, shape, scale=.1):
        weights[name] = (torch.randn(shape) * scale).bfloat16()

    random("model.embed_tokens.weight", (64, 64))
    random("lm_head.weight", (64, 64))
    weights["model.norm.weight"] = torch.ones(64).bfloat16()
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        for name in ["input_layernorm", "post_attention_layernorm"]:
            weights[prefix + name + ".weight"] = torch.ones(64).bfloat16()
        for name, shape in [("q", (128, 64)), ("k", (64, 64)),
                            ("v", (64, 64)), ("o", (64, 128))]:
            key = prefix + "self_attn." + name + "_proj"
            random(key + ".weight", shape)
            random(key + ".bias", (shape[0],))
        random(prefix + "self_attn.sinks", (4,), 1.)
        random(prefix + "mlp.router.weight", (expert_count, 64))
        random(prefix + "mlp.router.bias", (expert_count,), 1.)
        for name, output_dim, input_dim in [("gate_up", 192, 64),
                                            ("down", 64, 96)]:
            key = prefix + "mlp.experts." + name + "_proj"
            shape = (expert_count, output_dim, input_dim // 32)
            weights[key + "_blocks"] = torch.randint(
                0, 256, (*shape, 16), dtype=torch.uint8)
            weights[key + "_scales"] = torch.randint(
                119, 123, shape, dtype=torch.uint8)
            if bias:
                random(key + "_bias", (expert_count, output_dim),
                       10. if name == "gate_up" else 1.)
    if selected_count == 1 and expert_count > 1:
        # All router logits tie: choose the lowest expert index.
        for layer in range(2):
            key = f"model.layers.{layer}.mlp.router."
            weights[key + "weight"].zero_()
            weights[key + "bias"].zero_()
    save_file(weights, str(path / "model.safetensors"))


def reference(path, count, layer_count=0):
    c = json.loads((path / 'config.json').read_text())
    w = Weights(path)
    h = c['head_dim']
    qh = c['num_attention_heads']
    kh = c['num_key_value_heads']
    selected = c['num_experts_per_tok']
    hidden = c['intermediate_size']
    dim = c['hidden_size']
    tokens = torch.tensor([(i * 7 + 1) % 32 for i in range(count)])
    x = w.get('model.embed_tokens.weight')[tokens]

    def norm(x, name):
        variance = x.square().mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + c['rms_norm_eps']) * w.get(name)
    freq = c['rope_theta'] ** (torch.arange(0, h, 2).float() / h)
    freq = 1 / freq
    phase = torch.arange(count).float()[:, None] * freq
    cos, sin = phase.cos(), phase.sin()

    def rope(a):
        a, b = a.chunk(2, dim=-1)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)
    layers = c['layer_types'][:layer_count or c['num_hidden_layers']]
    for l, kind in enumerate(layers):
        p = f'model.layers.{l}.'
        n = norm(x, p + 'input_layernorm.weight')

        def proj(name, heads):
            key = p + 'self_attn.' + name + '_proj'
            value = F.linear(n, w.get(key + '.weight'), w.get(key + '.bias'))
            return value.reshape(count, heads, h).transpose(0, 1)
        q, k, v = (rope(proj('q', qh)), rope(proj('k', kh)), proj('v', kh))
        keys = k.repeat_interleave(qh // kh, 0).transpose(-1, -2)
        scores = (q @ keys) / math.sqrt(h)
        pos = torch.arange(count)
        mask = pos[None, :] <= pos[:, None]
        if kind == 'sliding_attention':
            mask &= pos[None, :] > pos[:, None] - c['sliding_window']
        scores = scores.masked_fill(~mask, -torch.inf)
        sinks = w.get(p + 'self_attn.sinks')
        if sinks is not None:
            sinks = sinks[:, None, None].expand(-1, count, 1)
            scores = torch.cat((scores, sinks), -1)
        prob = scores.softmax(-1)[..., :count]
        att = (prob @ v.repeat_interleave(qh // kh, 0)).transpose(0, 1)
        att = att.reshape(count, qh * h)
        key = p + 'self_attn.o_proj'
        x = x + F.linear(att, w.get(key + '.weight'), w.get(key + '.bias'))
        n = norm(x, p + 'post_attention_layernorm.weight')
        key = p + 'mlp.router'
        logits = F.linear(n, w.get(key + '.weight'), w.get(key + '.bias'))
        indices = logits.argsort(dim=-1, descending=True, stable=True)
        indices = indices[:, :selected]
        routing = logits.gather(1, indices).softmax(-1)
        gate = torch.empty(count, selected, hidden)
        up = torch.empty_like(gate)
        fc = torch.empty_like(gate)
        out = torch.empty(count, selected, dim)
        for xp in indices.unique().tolist():
            t, i = (indices == xp).nonzero(as_tuple=True)
            key = p + 'mlp.experts.'
            gu = F.linear(n[t], w.decode(key + 'gate_up_proj', xp),
                          w.get(key + 'gate_up_proj_bias', xp))
            gate[t, i], up[t, i] = (gu[:, ::2], gu[:, 1::2])
            g, u = (gu[:, ::2], gu[:, 1::2])
            limit = c.get('swiglu_limit', 0)
            if limit > 0:
                g, u = (g.clamp(max=limit), u.clamp(-limit, limit))
            act = g * torch.sigmoid(1.702 * g) * (u + 1)
            fc[t, i] = act
            down = F.linear(act, w.decode(key + 'down_proj', xp),
                            w.get(key + 'down_proj_bias', xp))
            out[t, i] = down * routing[t, i, None]
        ffn_out = torch.zeros_like(x)
        for i in range(selected):
            ffn_out += out[:, i]
        x = x + ffn_out
    final = F.linear(norm(x, 'model.norm.weight'), w.get('lm_head.weight'))
    return [
        ('norm', n[-1]), ('router', logits[-1]), ('indices', indices[-1]),
        ('routing', routing[-1]), ('gate', gate[-1]), ('up', up[-1]),
        ('fc', fc[-1]), ('weighted', out[-1]), ('ffn_out', ffn_out[-1]),
        ('logits', final),
    ]


def run(driver, *args):
    result = subprocess.run(
        [str(driver), *map(str, args)], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def compare(data, expected, label):
    offset = 0
    for name, ref in expected:
        dtype = np.uintp if name == "indices" else np.float32
        actual = np.frombuffer(data, dtype=dtype, count=ref.numel(),
                               offset=offset).reshape(ref.shape)
        offset += actual.nbytes
        target = ref.numpy()
        # Exact routing choices; FP32 reductions may round differently.
        matches = (actual == target if name == "indices" else
                   np.isclose(actual, target, atol=3e-4, rtol=2e-4))
        if not matches.all():
            index = tuple(int(i) for i in np.argwhere(~matches)[0])
            raise AssertionError(
                f"{label}: first divergence in {name}{index}: "
                f"C={actual[index]}, reference={target[index]}, "
                f"absolute error={abs(actual[index] - target[index])}")
    assert offset == len(data), "unexpected driver output size"


def check_decode(driver):
    data = run(driver, "decode")
    actual = np.frombuffer(data, dtype=np.float32).reshape(256, 16)
    values = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                       -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                      dtype=np.float32)
    with np.errstate(over="ignore"):
        expected = np.ldexp(values[None, :],
                            np.arange(255)[:, None] - 127)
    np.testing.assert_array_equal(actual[:255].view(np.uint32),
                                  expected.astype(np.float32).view(np.uint32))
    assert np.isnan(actual[255]).all(), "scale 255 must decode to NaN"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver", type=Path)
    parser.add_argument("--parallel", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    check_decode(args.driver.resolve())
    with tempfile.TemporaryDirectory(prefix="strasgpt-moe-") as directory:
        for experts, selected, bias, limit in [
                (1, 1, True, 7), (3, 2, True, 7),
                (3, 3, False, 7), (3, 1, True, 0)]:
            path = Path(directory) / f"{experts}-{selected}-{bias}-{limit}"
            fixture(path, experts, selected, bias, limit)
            counts = [9, 513] if (experts, selected) == (3, 2) else [9]
            for count in counts:
                for layer_count in [1, 2]:
                    expected = reference(path, count, layer_count)
                    baseline = None
                    chunks = [0, 1, 3] if count == 9 else [0, 127]
                    drivers = [(args.driver.resolve(), 1)]
                    if args.parallel:
                        drivers.append((args.parallel.resolve(), 10))
                    for driver, threads in drivers:
                        for chunk in chunks:
                            label = (f"{path.name}, tokens={count}, "
                                     f"layer={layer_count}, chunk={chunk}, "
                                     f"threads={threads}")
                            data = run(driver, path, count, chunk,
                                       layer_count, threads)
                            compare(data, expected, label)
                            if baseline is None:
                                baseline = data
                            assert baseline == data, label + ": outputs differ"
                    print(f"PASS {path.name}, tokens={count}, "
                          f"layers={layer_count}", flush=True)
    print("PASS MXFP4 decoding and MoE reference comparisons")


if __name__ == "__main__":
    main()
