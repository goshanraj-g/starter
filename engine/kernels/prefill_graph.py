"""Select a native BF16 BLAS backend once, then replay a fixed prefill graph."""

import statistics
import torch
from kernels.prefill import prefill


def graph_time(graph):
    graph.replay()
    times = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


class PrefillGraph:
    @torch.inference_mode()
    def __init__(self, model, cache, input_ids):
        self.input = torch.empty_like(input_ids)
        self.input.copy_(input_ids)
        self.cache = cache
        self.length = input_ids.shape[1]
        stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        candidates = []
        for backend in ('cublas', 'cublaslt'):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                cache.reset()
                logits = prefill(model, self.input, cache, check_rotary=True,
                                 blas_backend=backend)
                tokens = logits[:, -1].argmax(-1, keepdim=True)
            current.wait_stream(stream)
            del logits, tokens
            cache.reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                logits = prefill(model, self.input, cache, blas_backend=backend)
                tokens = logits[:, -1].argmax(-1, keepdim=True)
            current.wait_stream(stream)
            elapsed = graph_time(graph)
            candidates.append((graph, tokens, logits, elapsed))
        native, alternate = candidates
        choose_alternate = (
            alternate[3] < native[3] * 0.97
            and torch.equal(alternate[1], native[1])
            and torch.allclose(alternate[2], native[2], rtol=0.016, atol=0.125)
        )
        selected = alternate if choose_alternate else native
        self.graph, self.tokens = selected[:2]
        # Unselected graph pools are released when these local references die.

    def run(self, input_ids):
        self.input.copy_(input_ids)
        self.graph.replay()
        self.cache.length = self.length
        return self.tokens
