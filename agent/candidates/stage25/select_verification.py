"""Untimed whole-generation comparison; freeze the chosen backend per shape."""

from time import perf_counter
import torch

from kernels.cache import PrefixCache
from kernels.prefill_graph import PrefillGraph
from kernels.verify_graph import VerifyGraph, stream_verify


def select_verification(model, cache, prefiller, decoder, prompt, output_length, stream_decode):
    batch, length = prompt.shape
    total_memory = torch.cuda.get_device_properties(prompt.device).total_memory
    # The second graph/cache is temporary. Leave room for capture and model
    # workspaces, and do not specialize selection on benchmark identifiers.
    if (batch > 16 or length + output_length > 16384 or output_length < 4
            or 2 * torch.cuda.max_memory_allocated() + 8 * 2**30 > total_memory * .8):
        return None

    def ordinary(ids):
        cache.reset()
        first = prefiller.run(ids)
        host_first = first[:, 0].tolist()
        decoder.reset(first, length)
        return list(stream_decode(decoder, host_first, output_length))

    def timed(operation, ids):
        torch.cuda.synchronize()
        start = perf_counter()
        result = operation(ids)
        torch.cuda.synchronize()
        return perf_counter() - start, result

    # Run the reference before allocating speculative workspaces, preserving
    # the already-measured ordinary path's memory conditions.
    prompts = (prompt, prompt.roll(1, dims=1))
    reference = [timed(ordinary, ids) for ids in prompts]
    other_cache = PrefixCache(model.config, batch, length + output_length + 2,
                              prompt.device, model.dtype)
    other_prefiller = PrefillGraph(model, other_cache, prompt)
    first = other_prefiller.run(prompt)
    other_decoder = VerifyGraph(model, other_cache, prompt, first, output_length)

    def verified(ids):
        other_cache.reset()
        first = other_prefiller.run(ids)
        host_first = first[:, 0].tolist()
        other_decoder.reset(ids, first)
        return list(stream_verify(other_decoder, host_first, output_length))

    # Prime replay before timing. Check two prompts, including cache reset, and
    # require a material advantage on each rather than a lucky average.
    verified(prompt)
    chosen = True
    for ids, (baseline_time, baseline_output) in zip(prompts, reference):
        elapsed, result = timed(verified, ids)
        if result != baseline_output or elapsed >= baseline_time * .88:
            chosen = False
            break
    if chosen:
        return other_cache, other_prefiller, other_decoder, reference[0][1], True
    return cache, prefiller, decoder, reference[0][1], False
