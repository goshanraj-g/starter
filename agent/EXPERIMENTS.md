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

- Commit: `6c2c306c7599926c926e6ca3a8180c5ba4fa19fd`.
- Submission: `be0f94e3-e784-4102-b53b-bd2b3f022d02`.
- Run: `5620f648-e7e5-4b94-ba60-5bd98dfa0e30`; **passed every gate at
  747.8 tokens/s**, +1.0% versus stage 6.
- Raw report: `agent/results/stage7_prefill.json`.
- Public throughput: 192.6 / 393.6 / 2364.4 tokens/s.
- TTFT/native: 0.80 / 0.74 / 0.72; TPOT/native: 0.21 / 0.23 / 0.24.

- Calls pinned native dense FlashAttention with original Q/K/V projections,
  Q/K norms, and rotary. Preserves B,T,H,D strides and all initialized prompt
  keys; is_causal=True is valid because prompt positions start at zero.
- Writes each layer into persistent KV storage immediately, eliminating the
  duplicate DynamicCache and end-of-prefill copy loop.
- Applies final RMSNorm only to the last token, an independent row operation.
- CLI static validation passes; stage 6 passed; submitting prefill next.


## Stage 8 — measured small-batch BF16 split-K matrix products

- Commit: `4683b480945e1c245eb270a4c1d4c1e9bd7576fc`.
- Submission: `fbfdeee4-acd0-4666-b078-5119d68b3f16`.
- Run: `a52915a2-cce9-4ec8-bb78-4f1fb496075f`; **passed all gates at
  753.6 tokens/s**, +0.8% versus stage 7.
- Raw report: `agent/results/stage8_linear.json`.
- Public throughput: 193.5 / 398.1 / 2387.6 tokens/s.
- TTFT/native: 0.74 / 0.74 / 0.73; TPOT/native: 0.18 / 0.20 / 0.21.

- BF16 inputs/weights, FP32 tensor-core accumulation, FP32 partials and
  reduction, then a single BF16 output cast. No quantization or atomics.
- During warmup, compare one candidate output per operation against native,
  then GPU-time graph replays over all 36 distinct layer weights to avoid
  tuning only L2-resident microbenchmarks. Try 2/4/8 splits; LM head uses 1.
- Select Triton only with at least 10% measured improvement; otherwise retain
  cuBLAS. Batches above 32 retain native GEMM. Selection is fixed before the
  decode graph is captured; no tuning or CPU decisions during measured steps.
- CLI static validation passes. Stage 7 passed; submit the isolated matrix candidate next.


## Stage 11 — residual/RMSNorm and SwiGLU fusion

- Commit: `f4583d5c800fa5233568ef09f98a2122349678ce`.
- Submission: `18a3f273-5ec8-4b7f-bc9b-dd5f04e82445`.
- Run: `ee351947-5850-4811-b65e-808f2dd24169`; **failed unstable_timing**.
- Public correctness and latency checks passed. Public-1 median 311.7 ms,
  p10 309.5 ms, p90 433.8 ms, mean 351.7 ms, SD 91.2 ms. Other public
  cases were stable. No eligible hidden score.
- Raw report: `agent/results/stage11_pointwise.json`.
- Public throughput: 201.0 / 410.7 / 2566.0 tokens/s.
- TTFT/native: 0.79 / 0.75 / 0.72; TPOT/native: 0.15 / 0.17 / 0.18.

- Residual addition rounds to BF16 before norm statistics; FP32 normalization
  rounds to BF16 before learned gain, then BF16 output as in the reference.
- Carries the next layer's normalized input directly; final layer uses final
  model RMSNorm. Both residual branches remain in their original order.
- SwiGLU uses CUDA libdevice expf and round-to-nearest division, matching
  PyTorch's formula. SiLU rounds to BF16 before multiplication by up.
- Checked libdevice APIs against
  [Triton 3.1.0](https://raw.githubusercontent.com/triton-lang/triton/v3.1.0/python/triton/language/extra/cuda/libdevice.py).
- Prepared on top of grouped replay; CLI archive validation passes. Stage 10 passed; submit this isolated fusion next.


## Stage 9 — overlap GPU decode with token handoff

- Commit: `08ebd041c599d682e59afc79252c5b79510afc30`.
- Submission: `68190b8f-0b46-4cbb-bedb-c19627aa6abd`.
- Run: `d0d0a4ba-0f20-468f-ae4d-3999c3b526cd`; **passed every gate at
  775.6 tokens/s**, +2.9% versus stage 8.
- Raw report: `agent/results/stage9_overlap.json`.
- Public throughput: 197.3 / 398.4 / 2439.2 tokens/s.
- TTFT/native: 0.82 / 0.75 / 0.73; TPOT/native: 0.17 / 0.18 / 0.20.

- Copy the current token IDs to an owned host list, enqueue the next graph
  replay, then yield the host list. The harness can write the current output
  while the GPU decodes the following token. No math or kernel changes.
- Exactly output_length-1 graph replays; the final host copy waits for the last
  replay, so there is no outstanding generation work after the final yield.
- First-call graph capture stays in the platform's untimed warmup. Subsequent
  first-token handoffs add only an asynchronous launch, never a decode wait.
- A local control-flow test covers batch 1/4/16, output 2/32/128, immutable host
  snapshots, and exact replay counts. All three local tests pass.
- Stage 8 passed; submit this isolated scheduling change next.


## Stage 10 — grouped graph replay and host copies

- Commit: `ec87da983b4e10532132b7d1c7086f856a633098`.
- Submission: `ca987d41-e5be-4d2b-a0ed-29c6d3b351f0`.
- Run: `5493822d-88a3-4e02-b2b1-35714d62089e`; **passed every gate at
  787.3 tokens/s**, +1.5% versus stage 9.
- Raw report: `agent/results/stage10_grouped.json`.
- Public throughput: 199.1 / 403.3 / 2471.6 tokens/s.
- TTFT/native: 0.78 / 0.75 / 0.73; TPOT/native: 0.15 / 0.18 / 0.18.

- Capture up to eight sequential, exact single-token decode steps per graph,
  recording each token to a persistent [group,B] int64 output buffer.
- Copy one output group to host, enqueue the following group, then yield every
  saved list in order. The prefill token is yielded before waiting for any
  decode group; TTFT adds only the asynchronous graph enqueue.
- Capture both full and tail group sizes during warmup in independent pools.
  Include output length in the graph/cache reuse key. No graph computes extra
  tokens past the requested length. No outstanding GPU work after final yield.
- Local streaming test includes partial groups and exact multiples (output
  2/8/9/10/32/128, B 1/4/16), host ownership, order and total step count.
- All three local tests and CLI archive checks pass. Stage 9 passed; submit grouped replay next.


## Stage 12 — wider GEMM tiles and direct small-batch GEMV

- Commit: `2532e1ff7cc6922f80a629c1ef266895109f6f53`.
- Submission: `6bcab0c7-a6ad-4d7a-853d-b65e8d31d6f0`.
- Run: `be613823-a15b-4c57-97e9-9849f92e2758`; **passed every gate at
  815.5 tokens/s**, +0.3% versus stage 11a.
- Raw report: `agent/results/stage12_wide.json`.
- Public throughput: 209.2 / 421.9 / 2571.2 tokens/s.
- TTFT/native: 0.77 / 0.74 / 0.72; TPOT/native: 0.17 / 0.18 / 0.20.
- Revert this broader search: tiny score change does not justify a >12-minute
  run and additional compilation. Stage 13 uses the stable stage 11a matrix
  selection, so its comparison base is 813.4 tokens/s.

- Adds 128-column tensor-core tiles with K=64/128 to warmup comparisons.
- Adds direct FP32 product/reduction over BF16 inputs and weights for B<=4.
  These are exact matrix products with different reduction ordering, retaining
  BF16 outputs. Every candidate is compared against native before timing.
- Measures complete layer weight sets and requires a 5% improvement before
  selection. All choices remain fixed across the five measured samples.
- CLI archive validation passes; stage 11a passed; submitting the matrix comparison next.


## Stage 11a — prime prefill allocations after graph capture

- Commit: `25da6d08aeca309dbbb4337183cf2d8483f8696a`.
- Submission: `c876d589-34db-41fe-b8dc-24394d484df8`.
- Run: `3923ec83-f9cc-46fb-8caa-fd39b68f4d2d`; **passed every gate at
  813.4 tokens/s**, +3.3% versus the last eligible stage 10.
- Raw report: `agent/results/stage11a_primed.json`.
- Public throughput: 205.8 / 415.5 / 2572.7 tokens/s.
- TTFT/native: 0.77 / 0.74 / 0.72; TPOT/native: 0.16 / 0.18 / 0.19.

- Code review found that `torch.cuda.graph` clears allocator caches on entry;
  our decode captures occur after warmup prefill, leaving its eager allocations
  cold for the first measured sample. The aggregate timing report cannot prove
  this caused the outlier, but it is a concrete first-sample cost to eliminate.
- After initial graph construction, repeat prefill once during untimed warmup.
  Reset logical cache length and recompute the first token before graph reset.
  Measured calls do not repeat prefill or capture.
- Keep stage 12 matrix changes unsubmitted until this stability fix is measured.


## Stage 13 — packed QKV and fused normalization/rotary for prefill

- Commit: `c3d11bcc2ddc9cbda787f3175abb1faab160b89b`.
- Submission: `5ef1b230-455c-4410-847e-b11d8b7a7e21`.
- Run: `8f71648b-6eff-42f2-aaea-969d539940b6`; **failed incorrect_output**
  on a hidden case. Reverted entirely.
- Raw report: `agent/results/stage13_prefill_fused.json`. Public correctness
  passed, but that does not establish hidden correctness.
- Public throughput: 218.3 / 450.9 / 2672.9 tokens/s.
- TTFT/native: 0.53 / 0.63 / 0.61; TPOT/native: 0.14 / 0.16 / 0.17.
- Logs suppress hidden-workload details, so they do not identify the bad
  position. Packed prefill projection and extended fusion were changed
  together; neither is retained without a separate numerical verification.

- Extends the tested Q/K epilogue over [B,T] rows. COS/SIN index prompt token
  positions; K/V writes token-major cache positions 0..T-1. Decode retains
  device-side absolute positions for T=1.
- Uses the existing packed BF16 QKV projection for prefill, then one epilogue
  instead of separate head norms, rotary intermediates, and cache copies.
- Native causal GQA FlashAttention reads the initialized prefix views directly.
  The FP32/BF16 cast boundaries remain identical to the tested decode kernel.
- Static archive checks pass. Stage 12 measured and reverted; submit prefill fusion next.


## Stage 14 — CUDA graph prefill

- Commit: `571f9dd4c2122176ccd121ed3256e190c0357894`.
- Submission: `8ec161e2-1c2e-452d-91e8-919a307f40de`.
- Run: `f58724d0-7750-45a3-8be5-cc0ec25ea156`; **passed all gates, 811.9 tokens/s**.
- Raw report: `agent/results/stage14_prefill_graph.json`.
- Public throughput: 213.5 / 411.3 / 2553.9 tokens/s.
- TTFT/native: 0.34 / 0.74 / 0.73; TPOT/native: 0.11 / 0.12 / 0.13.
- Peak memory: 16.36 GB. Public batch-1 TTFT fell from 23.1 to 15.9 ms;
  larger-batch prefill and overall hidden throughput were effectively unchanged.
- Retain graph prefill for its small-batch latency benefit and persistent buffers.

- Capture fixed-shape prefill, final norm/head, and greedy argmax with persistent
  input and token buffers. Warm native/Triton paths on the capture stream first.
- Replays overwrite the full prompt prefix; shape changes release both prefill
  and decode graphs before reallocating cache. Copy current prompt IDs in place.
- Prefill graph pools keep their allocations alive, eliminating the cold eager
  allocator risk. Remove the now-unneeded repeated eager prefill warmup.
- First token is still handed off before waiting for any decode group.
- Uses the previously passing native GQA prefill path; stage 13 fusion is
  reverted. No prefill arithmetic or projection reformulation is included.
- CLI archive validation and three local tests pass; submit this graph-only
  prefill optimization against the stable stage 11a base.


## Stage 15 — measured column-major BF16 decode weights

- Commit: `e58c8ad1c1ad3964ea5d7521ba4c196c5a9d6545`.
- Submission: `f3df10a9-93fa-4b64-9807-93b60314dd05`.
- Run: `92fada53-4134-4bd5-869e-e91f2870af3f`; **passed all gates, 819.3 tokens/s**.
- Raw report: `agent/results/stage15_layouts.json`.
- Public throughput: 224.1 / 414.7 / 2550.2 tokens/s.
- TTFT/native: 0.57 / 0.74 / 0.72; TPOT/native: 0.19 / 0.22 / 0.22.
- Peak memory: 16.52 GB. Retain the measured layout selection.

- Compare native row-major weights with a column-major BF16 copy for each
  decode projection category. Values and formula stay unchanged; prefill
  retains original weights and layout.
- Profile all layer weights during warmup, compare outputs against native,
  and retain alternate storage only when it wins by at least 5%.
- Bound baseline peak plus retained/candidate copies to 85% of device memory,
  leaving room beneath the 90% gate for graph buffers.
- CLI archive validation and three local tests pass. Submitting after stage 14
  passed; H100 numerics and timing remain remote-only checks.


## Stage 16 — measured full-prefix Triton decode attention

- Commit: `d0fb1172fa469ed4c873724ac56552b30c7ff24b`.
- Submission: `4cbe542f-52b4-48b9-98b8-0155890261cf`.
- Run: `ac774750-1527-4c30-81fe-f54b3dc900d9`; **passed all gates, 872.5 tokens/s**.
- Raw report: `agent/results/stage16_attention.json`.
- Public throughput: 220.3 / 435.7 / 2726.7 tokens/s.
- TTFT/native: 0.52 / 0.73 / 0.72; TPOT/native: 0.16 / 0.16 / 0.16.
- Peak memory: 16.52 GB. Retain: hidden score improves 6.5%.

- Compare native variable-length FlashAttention with two exact full-prefix
  split-K candidates: scalar query-head reduction and grouped-query tensor cores.
- BF16 Q/K/V, FP32 online softmax/accumulators, BF16 probability products,
  FP32 partition combination, BF16 output. Query head h maps to KV head h//4.
- Device-side valid lengths mask every load and score; no eviction or pruning.
- Warmup checks short, middle and full prompt prefixes against native attention
  on first/last layer caches. Profile all 36 cache layers, retain a candidate
  only with a measured 10% advantage, then capture a fixed choice.
- Stage 15 passed. Local syntax/archive checks pass but do not establish
  numerical correctness or performance on H100.
- Checked the online-softmax and probability-cast structure against Triton
  3.1.0's official tutorial:
  https://raw.githubusercontent.com/triton-lang/triton/v3.1.0/python/tutorials/06-fused-attention.py
- Partition/reduction reference: https://pytorch.org/blog/flash-decoding/ .
  This candidate uses BF16 rather than the tutorial's FP16; no FP8 path is used.


## Stage 17 — rotary-only prefill fusion

- Keep separate native Q/K/V projections and the proven per-head norms.
- Fuse only BF16 rotary products/addition and full-prefix K/V cache writes.
- Eager warmup checks bitwise equality against native Q/K rotary and V cache
  contents at every layer. Capture then performs no host-side checks.
- No clear indexing bug was found in the failed stage 13 combined fusion;
  its packed prefill projections and fused normalization remain reverted.
- Prepared in the local engine, with drafts under `agent/candidates/`.
  Archive validation passes. Submitting after stage 16 passed all gates.


## Prepared stage 18 — parallel small-batch matrix-vector reduction

- Draft `agent/candidates/matvec_split.py` divides K into 1024-element chunks
  computed independently, using BF16 operands and FP32 products/reductions.
- A second kernel sums FP32 partials and performs the single output BF16 cast.
  No sequential K loop and no padded-partial copy/zero kernels.
- Intended for the existing measured small-batch selector, with its numerical
  checks and minimum improvement threshold. Not imported or submitted.
