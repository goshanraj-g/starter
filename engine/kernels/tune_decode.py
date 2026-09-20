"""Warmup-only full-decode check of per-operation matrix choices."""

import statistics
import torch
from torch.nn.functional import linear as native_linear


def group_time(decoder, first_token, first_position):
    count = decoder.chunk_size
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        decoder.reset(first_token, first_position)
        for slot in range(count):
            decoder.step()
            decoder.output[slot].copy_(decoder.tokens[:, 0])
    current.wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        decoder.reset(first_token, first_position)
        for slot in range(count):
            decoder.step()
            decoder.output[slot].copy_(decoder.tokens[:, 0])
    current.wait_stream(stream)
    graph.replay()
    times = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    result = decoder.output[:count].clone()
    del graph
    return statistics.median(times), result


@torch.inference_mode()
def tune_decode(decoder, first_token, first_position):
    choices = [(name, operation) for name, operation in decoder.linears.items()
               if operation is not native_linear]
    if not choices:
        return
    best_ms, reference = group_time(decoder, first_token, first_position)
    for name, operation in choices:
        decoder.linears[name] = native_linear
        elapsed, actual = group_time(decoder, first_token, first_position)
        if elapsed < best_ms * 0.98 and torch.equal(actual, reference):
            best_ms = elapsed
        else:
            decoder.linears[name] = operation
    decoder.reset(first_token, first_position)
