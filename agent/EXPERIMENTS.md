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
The candidate introduces no Triton or CUDA-graph API yet.

Next: collect the unchanged baseline's report. Its source
is preserved under `agent/reference/`. Run
`python3 agent/verify_engine.py MODEL_PATH` when a pinned H100 runtime is
accessible, obtain a measured stage-1 report through the connected repository,
record every latency gate and correctness result, then consider CUDA graphs.
