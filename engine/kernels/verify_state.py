"""Commit only model-verified tokens; propose from previous context matches."""

import triton
import triton.language as tl


@triton.jit
def _commit(PREDICTIONS, INPUT, POSITION, LENGTH, COUNT, OUTPUT, HISTORY,
            PROMPT: tl.constexpr, OUTPUTS: tl.constexpr, CAPACITY: tl.constexpr):
    batch = tl.program_id(0)
    count = tl.load(COUNT + batch)
    p0 = tl.load(PREDICTIONS + batch * 2)
    p1 = tl.load(PREDICTIONS + batch * 2 + 1)
    draft = tl.load(INPUT + batch * 2 + 1)
    accepted = tl.minimum(1 + (draft == p0).to(tl.int32), OUTPUTS - count)
    active = accepted > 0
    tl.store(OUTPUT + batch * OUTPUTS + count, p0, active)
    tl.store(OUTPUT + batch * OUTPUTS + count + 1, p1, accepted == 2)
    tl.store(HISTORY + batch * CAPACITY + PROMPT + count, p0, active)
    tl.store(HISTORY + batch * CAPACITY + PROMPT + count + 1, p1, accepted == 2)
    tl.store(INPUT + batch * 2, tl.where(accepted == 2, p1, p0), active)
    tl.store(INPUT + batch * 2 + 1, p1, active)
    position = tl.load(POSITION + batch) + accepted
    tl.store(POSITION + batch, position)
    tl.store(LENGTH + batch, position + 2)
    tl.store(COUNT + batch, count + accepted)


@triton.jit
def _propose(INPUT, COUNT, HISTORY, PROMPT: tl.constexpr, OUTPUTS: tl.constexpr,
             CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    count = tl.load(COUNT + batch)
    length = PROMPT + count
    history = HISTORY + batch * CAPACITY
    last0 = tl.load(history + length - 3, length >= 3, -1)
    last1 = tl.load(history + length - 2, length >= 2, -1)
    last2 = tl.load(history + length - 1)
    index = tl.arange(0, BLOCK)
    valid = (index >= 2) & (index < length - 1) & (length >= 3)
    a = tl.load(history + index - 2, valid, -2)
    b = tl.load(history + index - 1, valid, -2)
    c = tl.load(history + index, valid, -2)
    match = valid & (a == last0) & (b == last1) & (c == last2)
    latest = tl.max(tl.where(match, index, -1), 0)
    proposal = tl.load(history + latest + 1, latest >= 0, 0)
    tl.store(INPUT + batch * 2 + 1, proposal, (latest >= 0) & (count < OUTPUTS))


def propose(decoder):
    _propose[(decoder.batch,)](
        decoder.tokens, decoder.count, decoder.history, decoder.prompt_length,
        decoder.output_length, decoder.history.shape[1],
        triton.next_power_of_2(decoder.history.shape[1]), num_warps=4,
    )


def commit(decoder, predictions):
    _commit[(decoder.batch,)](
        predictions, decoder.tokens, decoder.position, decoder.valid_lengths,
        decoder.count, decoder.output, decoder.history, decoder.prompt_length,
        decoder.output_length, decoder.history.shape[1], num_warps=4,
    )
    propose(decoder)
