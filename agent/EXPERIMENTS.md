# Optimization record

## 2026-09-19 — stage 1, awaiting GPU validation

- Read the contract, AGENTS.md, optimization guide, baseline, and RMSNorm example in full.
- Preserved the original baseline in `agent/reference/baseline_engine.py`.
- Moved the unused RMSNorm example to `agent/reference/rmsnorm.py`; the
  submitted tree contains only imported modules.
- Candidate bypasses the top-level Transformers wrapper, retaining native
  decoder layers, norms, RoPE, projections, SDPA, and argmax.
- Reuses preallocated per-layer KV storage, exposes only initialized prefixes,
  supplies an explicit causal mask, and resets cache length on every generation.
- This is an eager cache, not yet graph compatible. No later stage is justified
  until stage 1 has numerical and timing measurements.
- `agent/verify_engine.py` checks the exact runtime, prefill and four cached
  decode forwards, and two different consecutive prompt streams for each public
  shape against separate, untouched native model weights. Local test prompts
  are synthetic prose; they do not reproduce the benchmark corpus.
- Local environment: Python 3.12.3, no torch/triton/transformers, no visible
  NVIDIA device. Numerical comparison and GPU timing cannot run here.
- CLI was downloaded with the official installer and checksum verified.
  Version reports 0.1.0; its help still describes public runs and uploads.
  The current upstream installer is identical to this repository's installer.
- Authenticated `dryft doctor` succeeds with an explicit
  `DRYFT_API=https://htn.dryft.ai`; the CLI default returned HTTP 403.
  Team submissions and runs are both empty. Credentials are held only in the
  tool shell's environment, outside source and archives.
- The authenticated challenge confirms six hidden workloads and the pinned
  checkpoint, runtime, 2-logit margin, and 1.10 latency limits. Its descriptive
  text still mentions public runs; follow the live participant workflow instead.
- GitHub authentication was subsequently supplied. Push access works, and
  a push to `main` automatically created a Dryft submission and official run,
  confirming the repository connection.
- Local validation: CLI archive validation passed; every engine and agent
  Python source parses for Python 3.11; both existing client unit tests passed;
  `git diff --check` passed. These checks do not validate GPU numerics.
- Packaged `submission.tar.gz`: 1,859 bytes, SHA256
  `697a28b01a849ea37ee9d398e3b12602d5226c0ecfac188e939ea7953c03395d`.
- Throughput delta, TTFT/TPOT ratios, memory, spread, and hidden score: **not measured**.
  The candidate has not been submitted. The 1000 tokens/s goal remains unverified.

## 2026-09-19 — unchanged baseline run

- Empty trigger commit: `d0ebca44448393e12d5912fbf9ba6b67207f68cd`.
- Submission: `44882493-4cfc-46b5-b243-6419b2e37155`.
- Official run: `7e366d00-be71-4a12-b782-ce78da0067ff`.
- Engine in this commit is the untouched repository baseline. The stage-1
  candidate remains uncommitted while the baseline is measured.
- Result: succeeded and ranked, **192.2977 tokens/s** hidden geometric mean.
- Public 0/1/2 throughput: 43.4248 / 135.5267 / 637.7393 tokens/s.
- Public TTFT ratios: 1.0668 / 1.0014 / 1.0052; TPOT ratios:
  1.0426 / 0.9889 / 1.0099. All cases passed.
- Whole-run reported peak: 14,484,570,112 bytes. Public per-case peak:
  10,275,651,584 / 12,280,463,360 / 12,280,463,360 bytes.
- Raw report: `agent/results/baseline.json`. Private per-case details and
  exact sample-spread values are not exposed in this report.
- Completed in approximately nine minutes on H100 80GB; Python 3.11.5.

## 2026-09-19 — stage 1 submission

- Commit: `ff6e5194137537b7c56638d68c16f0f362b18f90`.
- Submission: `c28735a5-b722-4fca-88c8-588d31e2a6d9`.
- Official run: `777fc681-d806-4741-962e-78f61bf33eb9`.
- Initial state: queued, awaiting a GPU.
- Archive lint passes. Local GPU numerical validation remains unavailable;
  correctness and timing will be evaluated by the platform.
- Changes are limited to wrapper bypass, prefix views over preallocated KV
  storage, explicit causal masks, and reset on each generation.
- Awaiting platform report before proceeding to CUDA graphs.

### Stage 1 result and isolated prefill correction

- Failed `latency_limit`. All three exposed public cases passed correctness.
- Public 0/1/2 throughput: 54.1 / 144.1 / 712.7 tokens/s. These cross-run
  changes cannot alone establish a speedup because native timings also shifted.
- Public TTFT/native: 1.01 / **1.28** / 1.06. TPOT/native: 1.05 / 0.98 / 0.93.
- Raw report: `agent/results/stage1.json`. No eligible hidden score.
- Likely regression: explicit prefill mask changes SDPA's causal dispatch.
- Correction retains native empty DynamicCache causal prefill with no mask,
  then copies its KV tensors into preallocated storage. Cached forwards retain
  the initialized-prefix view and explicit mask. No CUDA graphs or fused ops
  added yet, to isolate the prefill change.
- Prefill-correction commit: `95b0bcd457333ff8d82fb616d53c78ce60ef7ba2`.
- Run: `235eeb2c-8c59-46a6-9d64-1256c79430ca`; passed and ranked at
  **118.5926 tokens/s**. Raw report: `agent/results/stage1_prefill.json`.
- TTFT/native: 1.034 / 1.010 / 1.006; TPOT/native: 1.062 / 1.006 / 1.059.
  This fixed prefill latency but did not demonstrate a decode speedup.
- Native TPOT in this run was 36.8–39.9 ms versus 17.4–22.5 ms in the
  previous run. Compare paired ratios; the raw score drop alone cannot be
  attributed to the code. Peak memory: 17,051,484,160 bytes.

## Stage 2 — CUDA graph decode

- Commit: `68c91155964d4ceba602cfc6e7ca60709e5c3ba9`.
- Submission: `56d521b8-5821-4b61-8a24-b693bde39def`.
- Run: `e25b30d7-452f-4349-b564-183b3c4f3ab4`; failed latency gate.
- Raw report: `agent/results/stage2_graph.json`. Public correctness passed.
- Public 0/1/2 throughput: 95.2 / 168.6 / 777.7 tokens/s.
- TPOT/native: **0.38 / 0.62 / 0.68**, demonstrating decode improvement.
- TTFT/native: **1.20** / 1.02 / 1.01. Public-0 first-token latency failed;
  graph capture alone does not reduce eager prefill work. No eligible score.
- Reviewed the [PyTorch 2.5.1 graph implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/cuda/graphs.py)
  for explicit capture streams, warmup, replay, and buffer lifetime.
- Added a fixed-capacity decode cache with GPU `index_copy_`, a GPU absolute
  position, and a boolean valid-prefix mask. Unused storage is initialized to
  zero once to avoid NaNs in masked value lanes.
- Captures the complete native single-token forward, argmax, next-token update,
  and position increment. Three side-stream warmup steps precede capture.
- Every generation resets input IDs/position and overwrites prompt cache slots.
  Host token conversion and per-step yield remain outside the graph.
- Graph capture is per storage shape during the first warmup generation;
  subsequent same-shape prompts replay without capture or recompilation.
- Native causal prefill remains unchanged from the isolated prefill correction.
- Archive lint and Python 3.11 parsing pass. GPU validation remains remote.
- Research follow-on: [Flash-Decoding](https://pytorch.org/blog/flash-decoding/)
  parallelizes attention over KV sequence partitions and combines normalized
  partial results using log-sum-exp. Consider only after measuring graph decode.

## Stage 3 — fused normalization

- Commit: `dd59b5504228982e5fff9a58746256c60c367b60`.
- Submission: `850205e9-eaa2-442c-b63f-6127e98e61ff`.
- Run: `fb1f2514-8559-479f-9b00-2645433864f3`; **passed and ranked at
  339.7 tokens/s**, with all gates passed.
- Raw report: `agent/results/stage3_norm.json`.
- Public throughput: 120.9 / 197.6 / 905.9 tokens/s.
- TTFT/native: 0.89 / 0.81 / 0.80; TPOT/native: 0.28 / 0.53 / 0.64.
- Reinstated the provided Triton RMSNorm implementation as an imported module.
- Replaces hidden, per-head Q/K, and final norms; weights and epsilon are reused.
- Preserves FP32 reduction/normalization followed by BF16 cast before multiplying
  by the learned weight. No rotary, residual, attention, or MLP formula changes.
- Applies to prefill and captured decode, targeting the first-token regression
  as well as repeated GPU operations within the graph.
- Archive lint passes. Numerical validation remains remote; the local harness
  compares against a separate untouched baseline model when H100 access exists.

## Stage 4 — native variable-length grouped-query FlashAttention

- Commit: `42f2df5f225eabb86ec3c178beb23dc55adeb21b`.
- Submission: `f02f0629-8eff-4de3-a03e-ffa4ded863b8`.
- Run: `57019eb9-21ae-4aed-9940-e2cee6f4a8e3`; **passed all gates, ranked
  at 600.5 tokens/s** (+76.8% versus stage 3).
- Raw report: `agent/results/stage4_flash.json`.
- Public throughput: 152.5 / 333.3 / 1889.3 tokens/s.
- TTFT/native: 0.87 / 0.81 / 0.79; TPOT/native: 0.35 / 0.34 / 0.32.
- Research found a suitable native operator already in the pinned runtime:
  `aten._flash_attention_forward(..., seqused_k=...)`.
- Checked its exact [2.5.1 schema](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/native_functions.yaml)
  and [CUDA dispatch](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/transformers/cuda/attention.cu).
- The [pinned FlashAttention implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp)
  verifies contiguous int32 CUDA cumulative offsets and valid lengths. For
  one-query GQA it groups queries by KV head and enables split-K decoding.
- Physical KV layout becomes [B, capacity, 8, 128], exposing the same logical
  [B, 8, capacity, 128] views. Flattened token-major storage gives the operator
  fixed reserved batch segments without a per-step transpose or KV repetition.
- Query offsets are [0, 1, ..., B]; key offsets are [0, C, ..., B*C]. Separate
  valid lengths start at prompt_length + 1 and advance on the GPU with position.
- Direct decoder steps preserve native Q/K norms, rotary function, projections,
  both BF16 residual additions, SwiGLU, and the tied LM head.
- No custom attention math or newer-release API. Initial archive/syntax checks
  pass. Normalization passed; submitting this attention change next.

## Live workflow supersedes the repository's older run instructions

[Live docs](https://htn.dryft.ai/docs), fetched 2026-09-19, specify official
runs only, six hidden workloads, and new submissions via connected GitHub
repositories. Do not call the old public-run or archive-upload workflow.
The user-provided signed-in docs additionally report a 15-minute whole-run
limit; the anonymous docs do not expose the current live limit.

Pinned API review:
[Qwen3 4.51.3](https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)
passes the supplied cache to each attention layer's `update` method; direct
decoder-layer calls avoid the model wrapper's cache type check.
The unchanged baseline source is preserved under `agent/reference/`. Run
`python3 agent/verify_engine.py MODEL_PATH` when a pinned H100 runtime is
accessible. This workspace has no GPU or pinned numerical libraries; static
validation here does not establish numerical correctness.


## Stage 5 — packed decode projections

- Commit: `0ec1f0f249343f87cbc9c7b6d96e3802724f1e09`.
- Submission: `70817beb-2d94-4dbb-962b-4017dc1578b3`.
- Run: `afd4ffd8-a591-47b4-a06f-9ccde35ad67f`; **passed all gates at
  609.7 tokens/s** (+1.5% versus stage 4).
- Raw report: `agent/results/stage5_packed.json`.
- Public throughput: 155.4 / 337.3 / 1919.1 tokens/s.
- TTFT/native: 0.86 / 0.80 / 0.80; TPOT/native: 0.22 / 0.24 / 0.28.

- Concatenate Q/K/V projection rows and gate/up projection rows once during
  model loading. Rebind native Linear weights to contiguous slices sharing the
  packed allocations, preserving prefill without duplicating model weights.
- Decode uses four matrix products per layer instead of seven: QKV, output,
  gate/up, and down. Native SiLU, multiplication, norm, and rotary remain.
- All weights, intermediate outputs, and residuals remain BF16. Packing may
  select different cuBLAS tiling, so the platform must check correctness.
- CLI archive validation passes. Stage 4 passed; submit this isolated projection change next.
- Further fusion research: pinned PyTorch
  [SiLU kernel](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/cuda/ActivationSiluKernel.cu)
  computes x/(1+exp(-x)) in opmath precision and returns the input scalar dtype.
  Any SwiGLU fusion must round SiLU to BF16 before multiplying by up.


## Stage 6 — fused Q/K normalization, rotary, and KV write

- Commit: `333ec9cceb943635ae5d2d93bcfdb1f70aa42ab0`.
- Submission: `b933a32b-d362-411b-b526-30dcddefe214`.
- Run: `bba74193-6619-474c-8a6b-817223376245`; **passed every gate at
  740.3 tokens/s**, +21.4% versus stage 5.
- Raw report: `agent/results/stage6_qkv.json`.
- Public throughput: 192.6 / 379.9 / 2330.0 tokens/s.
- TTFT/native: 0.93 / 0.81 / 0.80; TPOT/native: 0.22 / 0.24 / 0.25.

- One Triton launch replaces two per-head norms, native rotary elementwise
  operations, layout copies, and two cache writes for each decode layer.
- Inputs remain packed BF16 QKV; output Q is contiguous [B,32,128]. K/V write
  only the current absolute position in the token-major cache.
- FP32 norm reduction; BF16 rounding after normalization, learned gain, each
  cosine/sine product, and rotary sum. V is copied without normalization.
- `enable_fp_fusion=False` is supported by the pinned
  [Triton 3.1.0 backend](https://raw.githubusercontent.com/triton-lang/triton/v3.1.0/third_party/nvidia/backend/compiler.py).
- Projection diagnostic remains outside engine: successful official-run logs
  explicitly suppress candidate output once hidden workloads are touched, so
  those measurements cannot currently be inspected. Do not ship unused logging.
- CLI static validation and Python 3.11 syntax pass. Not submitted; awaiting
  stage 6 remote measurement.


## Stage 7 — native grouped-query causal prefill

- Calls pinned native dense FlashAttention with original Q/K/V projections,
  Q/K norms, and rotary. Preserves B,T,H,D strides and all initialized prompt
  keys; is_causal=True is valid because prompt positions start at zero.
- Writes each layer into persistent KV storage immediately, eliminating the
  duplicate DynamicCache and end-of-prefill copy loop.
- Applies final RMSNorm only to the last token, an independent row operation.
- CLI static validation passes; stage 6 passed; submitting prefill next.
