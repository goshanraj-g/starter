"""Offline H100 check: python3 agent/verify_engine.py /path/to/checkpoint.

Uses the unchanged baseline, separate model instances, all public shapes,
prefill/cached logits, and consecutive prompts. This is a local correctness
check, not a replacement for the platform's timing and teacher-forced replay.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((1, 512, 32), (4, 2048, 32), (16, 512, 128))


def check_runtime():
    expected = {
        "torch": "2.5.1", "triton": "3.1.0", "transformers": "4.51.3",
        "safetensors": "0.5.3", "tokenizers": "0.21.1",
    }
    errors = []
    if sys.version_info[:2] != (3, 11):
        errors.append(f"Python 3.11 required; found {sys.version.split()[0]}")
    for name, version in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual.split("+")[0] != version:
            errors.append(f"{name} {version} required; found {actual}")
    if errors:
        raise SystemExit("Runtime check failed:\n  " + "\n  ".join(errors))
    import torch
    if torch.version.cuda != "12.4" or not torch.cuda.is_available():
        raise SystemExit("CUDA 12.4 and an accessible H100 are required")
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise SystemExit("This check targets exactly one visible H100")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--corpus", type=Path, help="Optional local UTF-8 prompt corpus")
    args = parser.parse_args()
    check_runtime()
    if args.runtime_only:
        print("Pinned H100 runtime verified")
        return
    if not args.model_path:
        parser.error("model_path is required unless --runtime-only is used")

    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "engine"))
    from engine import Engine, qwen_forward
    from kernels.cache import PrefixCache
    from reference.baseline_engine import Engine as Baseline

    candidate = Engine(args.model_path)
    baseline = Baseline(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    corpus = args.corpus.read_text() if args.corpus else (
        "Explain how a computer stores numbers and how arithmetic works. "
        "A garden contains apple trees, flowers and vegetables. Describe its seasons. "
        "Write a Python function to sort a list and explain its running time. "
        "Compare the properties of water, ice and steam. Give concrete examples. "
    )
    tokens = tokenizer.encode(corpus, add_special_tokens=False)
    if not tokens:
        parser.error("The prompt corpus produced no tokens")

    with torch.inference_mode():
        for batch, prompt, output in SHAPES:
            shape = [batch, prompt, output]
            prompts = [
                [[tokens[(j + i * 37 + sample * 101) % len(tokens)]
                  for j in range(prompt)] for i in range(batch)]
                for sample in range(2)
            ]
            # Independently compare prefill and four cached decode forwards.
            cache = PrefixCache(candidate.model.config, batch, prompt + output,
                                "cuda:0", torch.bfloat16)
            native_cache = None
            current = torch.tensor(prompts[0], device="cuda:0", dtype=torch.int64)
            for step in range(5):
                native = baseline.model(input_ids=current, past_key_values=native_cache,
                                        use_cache=True, logits_to_keep=1, return_dict=True)
                actual = qwen_forward(candidate.model, current, cache)
                expected = native.logits
                if not torch.isfinite(actual).all().item():
                    raise AssertionError(f"Nonfinite logits: {shape}, step {step}")
                delta = (actual.float() - expected.float()).abs().max().item()
                actual_ids = actual[:, -1].argmax(-1)
                expected_ids = expected[:, -1].argmax(-1)
                torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
                print(json.dumps({"shape": shape, "forward": step,
                                  "max_logit_delta": delta, "argmax_match": True}), flush=True)
                current = expected_ids[:, None]
                native_cache = native.past_key_values
            del cache, native_cache, native, actual, expected, current

            # Reuse the same Engine across different prompts and across shapes.
            for sample, ids in enumerate(prompts):
                expected_steps = list(baseline.generate(ids, output))
                actual_steps = list(candidate.generate(ids, output))
                if len(actual_steps) != output:
                    raise AssertionError("Wrong output step count")
                for step, (actual, expected) in enumerate(zip(actual_steps, expected_steps)):
                    if len(actual) != batch or any(type(token) is not int for token in actual):
                        raise AssertionError("Output must contain one Python int per sequence")
                    if actual != expected:
                        raise AssertionError(f"Greedy mismatch: {shape}, sample {sample}, step {step}")
                print(json.dumps({"shape": shape, "sample": sample,
                                  "greedy_stream_match": True}), flush=True)
    print("PASS: prefill, cached decode, and consecutive greedy streams for all public shapes")


if __name__ == "__main__":
    main()
