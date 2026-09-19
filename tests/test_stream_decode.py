"""Check streaming ownership and replay count without a local CUDA runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


source = Path(__file__).resolve().parents[1] / "engine" / "engine.py"
tree = ast.parse(source.read_text())
function = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "stream_decode")
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
stream_decode = namespace["stream_decode"]


class Tokens:
    def __init__(self, values):
        self.values = values[:]

    def __getitem__(self, index):
        return self

    def tolist(self):
        return self.values[:]


class StreamDecodeTest(unittest.TestCase):
    def test_host_snapshots_and_exact_replay_count(self):
        for batch in (1, 4, 16):
            for length in (2, 32, 128):
                first = list(range(batch))
                tokens = Tokens(first)
                replays = []

                def replay():
                    replays.append(True)
                    tokens.values = [value + 7 for value in tokens.values]

                decoder = SimpleNamespace(tokens=tokens, graph=SimpleNamespace(replay=replay))
                stream = stream_decode(decoder, first, length)
                self.assertEqual(next(stream), first)
                self.assertEqual(len(replays), 1)  # Next GPU step is already queued.
                result = [first] + list(stream)
                self.assertEqual(result, [[i + 7 * step for i in range(batch)]
                                          for step in range(length)])
                self.assertEqual(len(replays), length - 1)
                self.assertEqual(first, list(range(batch)))


if __name__ == "__main__":
    unittest.main()
