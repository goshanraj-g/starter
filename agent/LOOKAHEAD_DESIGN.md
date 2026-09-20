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

## Attention cost to resolve before trying it

Native FlashAttention's specialized single-query GQA path does not apply to
`T=2`. Use native causal varlen attention as the numerical reference, but a
Triton candidate may be necessary for any gain. The existing grouped-query
split kernel can represent `4*T` real rows per KV head in its padded 16-row tile.
For `T=2`, eight real rows still fit the same tensor-core tile as current decode.
Map row `r` to token `r//4` and head `kv_head*4 + r%4`; mask keys at or before
`valid_length - T + token_index` for each row. Read every valid prefix slot.

Unlike single-token attention, a nonempty partition for the second query may
be completely masked for the first. Handle each empty row explicitly: keep its
numerator/denominator zero and its maximum at negative infinity, avoiding
`exp(-inf - -inf)` NaNs. Store partials indexed by `[B,T,32,split,128]` and combine
as in the proven kernel. Compare short/full prefixes with the native reference.

At small batches, two-token GEMMs can remain within the same padded matrix
row tiles as single-token GEMMs. This is a performance hypothesis to measure,
not a guarantee. Native attention alone may erase the acceptance benefit.

## Conservative integration if cheaper options fail

Keep a separate speculative cache with capacity `S+O+2`, a separate prefill
graph, and a separate decode graph. Share immutable model weights. This lets the
ordinary decoder remain unchanged as a fallback and avoids changing its cache
capacity merely to test speculation. Release the unused decoder/cache/prefill
pool after selection.

Limit initial trials to small batches where two queries still fit the existing
16-row matrix tiles. During untimed warmup, compare whole generation (prefill,
all decode passes, and host snapshots) against the established decoder. Use
multiple trials, including a second same-shape prompt made by shifting the
warmup token IDs, to exercise cache reset and avoid choosing a path from one
favorable continuation. Require identical complete token sequences and a clear
whole-generation speed advantage. Keep that choice fixed in measured calls.

The first `generate` call can return the already computed outputs for its
original prompt after tuning; never return a timing trial's modified-prompt
outputs. Subsequent calls must reset and compute from their new input IDs.
Any warmup state from another trial is invalidated before the next call.

For `T=2`, the device update accepts `1 + (draft == prediction0)`, clamped to
remaining output count. Store only accepted predictions, update the current
input to the last accepted prediction, and reuse prediction1 as the next
unverified proposal. Completed rows freeze their inputs/position/count. Capture
1/2/4/8 passes; choose the largest power of two no greater than
`max(1, remaining//2)` and at most eight. This bounds wasted work near the tail.

Meaningful host tests should simulate prefix-dependent next-token oracles,
forced accept/reject patterns, overwritten invalid cache suffixes, differing
row progress, completed-row freezing, EOS IDs, and exact ordered yields for
short and long output counts. GPU attention/numerical verification remains
necessary; host simulations cannot validate Triton arithmetic.
