"""Exercise the actual streaming loop against a causal oracle and mutable KV."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


path = Path(__file__).resolve().parents[1] / "engine/kernels/verify_graph.py"
tree = ast.parse(path.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ("group_size", "stream_verify")]
namespace = {}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
stream_verify = namespace["stream_verify"]


class Matrix:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        rows, columns = key
        return Matrix([row[columns] for row in self.rows[rows]])

    @property
    def T(self):
        return Matrix([list(column) for column in zip(*self.rows)])

    def tolist(self):
        return [row[:] for row in self.rows]


class OracleDecoder:
    def __init__(self, oracle, proposal_mode):
        self.oracle = oracle
        self.mode = proposal_mode
        self.graphs = {n: SimpleNamespace(replay=lambda n=n: self.replay(n))
                       for n in (1, 2, 4, 8)}
        self.count = SimpleNamespace(tolist=lambda: self.progress[:])

    def reset(self, prompts, outputs):
        self.prompts = [p[:] for p in prompts]
        self.outputs = outputs
        self.progress = [1] * len(prompts)
        self.position = [len(p) for p in prompts]
        self.cache = [p[:] + [None] * (outputs + 2) for p in prompts]
        self.committed = [[self.oracle(p)] for p in prompts]
        self.output = Matrix([row[:] + [None] * (outputs - 1) for row in self.committed])
        self.inputs = [[row[0], row[0]] for row in self.committed]
        self.steps = 0
        self.accepted_two = 0
        self.rejected = 0
        self.frozen = 0
        self.propose()
        return [row[0] for row in self.committed]

    def propose(self):
        for b, prompt in enumerate(self.prompts):
            prefix = prompt + self.committed[b]
            prediction = self.oracle(prefix)
            if self.mode == "accept":
                self.inputs[b][1] = prediction
            elif self.mode == "reject":
                self.inputs[b][1] = prediction + 1
            elif self.mode == "mixed":
                self.inputs[b][1] = prediction + ((b + self.steps) % 3 == 0)
            elif self.mode == "uneven":
                self.inputs[b][1] = prediction + (b % 2)
            elif self.mode == "history":
                for end in range(len(prefix) - 2, 1, -1):
                    if prefix[end - 2:end + 1] == prefix[-3:]:
                        self.inputs[b][1] = prefix[end + 1]
                        break

    def replay(self, passes):
        for _ in range(passes):
            for b, prompt in enumerate(self.prompts):
                position = self.position[b]
                prefix = prompt + self.committed[b]
                if self.progress[b] == self.outputs:
                    self.frozen += 1
                    # A finished row may still execute ignored forwards but
                    # its state and emitted tokens cannot advance.
                    assert position + 1 < len(self.cache[b])
                    continue
                assert self.cache[b][:position] == prefix[:-1]
                current, proposal = self.inputs[b]
                assert current == prefix[-1]
                self.cache[b][position:position + 2] = [current, proposal]
                p0 = self.oracle(self.cache[b][:position + 1])
                p1 = self.oracle(self.cache[b][:position + 2])
                accepted = min(1 + (proposal == p0), self.outputs - self.progress[b])
                if accepted == 2:
                    self.accepted_two += 1
                elif proposal != p0:
                    self.rejected += 1
                predictions = [p0, p1][:accepted]
                start = self.progress[b]
                self.output.rows[b][start:start + accepted] = predictions
                self.committed[b].extend(predictions)
                self.progress[b] += accepted
                self.position[b] += accepted
                self.inputs[b] = [predictions[-1], p1]
            self.steps += 1
            self.propose()


class VerificationTest(unittest.TestCase):
    def test_exact_causal_outputs_and_two_resets(self):
        oracles = [lambda p: len(p) % 13,
                   lambda p: (p[-1] * 7 + 3) % 17,
                   lambda p: sum((i + 1) * x for i, x in enumerate(p)) % 19,
                   lambda p: 0]  # EOS-like ID must not stop generation.
        for oracle in oracles:
            for mode in ("accept", "reject", "mixed", "history"):
                decoder = OracleDecoder(oracle, mode)
                for batch in (1, 4, 16):
                    for length in (2, 3, 7, 8, 9, 32, 128):
                        for shift in (0, 11):
                            prompts = [[(b + i + shift) % 23 for i in range(5)]
                                       for b in range(batch)]
                            first = decoder.reset(prompts, length)
                            result = list(stream_verify(decoder, first, length))
                            reference = [p[:] for p in prompts]
                            expected = []
                            for _ in range(length):
                                tokens = [oracle(p) for p in reference]
                                expected.append(tokens)
                                for prefix, token in zip(reference, tokens):
                                    prefix.append(token)
                            self.assertEqual(result, expected)
                            self.assertEqual(decoder.progress, [length] * batch)
                            self.assertEqual(first, expected[0])
                            if mode == "reject":
                                self.assertEqual(decoder.steps, length - 1)
                            if mode == "accept" and length > 2:
                                self.assertGreater(decoder.accepted_two, 0)

    def test_mixed_rows_freeze_without_extra_outputs(self):
        decoder = OracleDecoder(lambda p: len(p) % 5, "uneven")
        first = decoder.reset([[1, 2, 3]] * 16, 32)
        result = list(stream_verify(decoder, first, 32))
        self.assertEqual(len(result), 32)
        self.assertGreater(decoder.accepted_two, 0)
        self.assertGreater(decoder.rejected, 0)
        self.assertGreater(decoder.frozen, 0)


if __name__ == "__main__":
    unittest.main()
