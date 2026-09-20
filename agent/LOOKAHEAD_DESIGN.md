# Unimplemented last-resort design: verified two-token Jacobi proposals

This is a proposed experiment, not a measured optimization or a claim that it
will improve this benchmark. Finish the cheaper kernel experiments first.

The primary reference is [Fu et al., ICML 2024](https://proceedings.mlr.press/v235/fu24a.html).
Their exact lookahead method extracts and verifies candidates using the target
model rather than an auxiliary checkpoint. The simpler rolling two-token design
below is an inference from that idea, not their full lookahead algorithm.

## Acceptance and cache invariant

A forward pass takes two inputs per sequence: the last committed token `a` and
an unverified proposal `d`. Its causal outputs are `p0` (the next-token argmax
after `a`) and `p1` (the next-token argmax after `a,d`).

- Always accept `p0`.
- Accept `p1` only when `d == p0`; its prefix is then exactly the committed one.
- If the proposal mismatches, discard the second cache slot. Overwrite it with
  `p0` before it can be read as valid history on the next forward pass.
- Reuse `p1` as the next unverified proposal. After accepting two tokens this
  starts as a repeated-token guess. It is never emitted without verification.
- Clamp accepted outputs to the remaining requested count. EOS stays ordinary.

Each sequence needs its own device position, valid length, and output count.
The model owns every emitted argmax; no proposal bypasses verification. Different
BF16 reduction orders still require the normal remote correctness gate.

## Graph-compatible state

Allocate inputs `[B,2]`, positions/valid lengths/counts `[B]`, and an output
buffer `[B,O]`. Initialize output slot zero with the prefill greedy token and
counts to one. Cache capacity is at least `S+O+2`, allowing a verification window
at the last output position without an out-of-range write.

Adapt the existing decode forward to two query tokens:

- Embed both input IDs; use native RoPE with position IDs
  `positions[:,None] + arange(2)[None,:]`.
- Extend the proven Q/K epilogue over `[B,T]`, reading the per-sequence position
  and the matching `[B,T,128]` cosine/sine row. Preserve every BF16 cast.
- Use native variable-length FlashAttention with `cu_query = arange(B+1)*2`,
  fixed-capacity `cu_key`, `seqused_k = positions+2`, and causal attention.
  Verify PyTorch 2.5.1's bottom-right causal alignment against actual valid
  lengths from the pinned source before implementation.
- Existing residual/norm and SwiGLU kernels already operate over arbitrary
  leading dimensions. Select matrix kernels for `B*2` rows.
- Apply the tied head to both positions and native lowest-index argmax.
- A device update kernel verifies the proposal, writes one or two committed
  outputs, advances position/count, and updates the next input/proposal.
- Freeze completed sequences' positions and inputs. Extra ignored forwards for
  such rows must remain within capacity and must never produce extra yields.

Capture groups of 1, 2, 4, and 8 verification passes. A group starts from current
device state; only warmup/timing probes reset it. Before each group, choose the
largest count at most `max(1, remaining//2)` so speculative progress does not
create a large wasted tail. Read output counts after replay, take the minimum
across sequences, snapshot newly available `[B]` output columns, queue the next
group, then yield those snapshots in order. Faster sequences can write ahead
but cannot change output ordering. Each unfinished row advances at least one
position per pass, ensuring termination.

## Required checks before retaining it

- Verify accept/reject transitions, differing per-sequence progress, discarded
  KV suffixes, final partial groups, exact yield count, and repeated generate
  calls using small deterministic host simulations.
- In untimed warmup, compare the complete generated sequence with the established
  single-token path, using the same prompt and a reset prefill cache.
- Time whole generation, including prefill and host snapshots, with both paths.
  Keep ordinary decode unless the speculative path wins by a substantial margin.
- Keep the backend fixed for measured samples. Prompt-dependent acceptance can
  still violate the 25% timing-spread gate, even if the warmup was fast.
- No separate weights, downloads, approximate attention, or unverified outputs.

The likely weakness is low acceptance for guesses derived from incorrect
prefixes, especially when the slowest sequence controls a batch's completion.
A larger verification window or the paper's n-gram lookahead branches could
improve proposals, but would add compute, masking complexity, and timing risk.
