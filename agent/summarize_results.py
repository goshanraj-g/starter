"""Summarize saved official reports without credentials or network access."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
reports = []
for path in (root / 'results').glob('*.json'):
    detail = json.loads(path.read_text())
    if 'result' not in detail:
        continue
    reports.append((detail.get('createdAt', ''), path, detail))
lines = ['# Official benchmark history', '',
         'Scores are hidden-workload geometric means. Failed runs are ineligible.', '',
         '| Experiment | State | Score (tok/s) | Public 0 / 1 / 2 (tok/s) | Peak GB |',
         '| --- | --- | ---: | --- | ---: |']
for _, path, detail in sorted(reports):
    result = detail.get('result') or {}
    public = sorted((s for s in result.get('shapes', []) if s.get('public')),
                    key=lambda s: s['id'])
    rates = ' / '.join(f"{s.get('tokensPerSecond', 0):.1f}" for s in public)
    score = result.get('score')
    score_text = f'{score:.1f}' if score is not None else '—'
    peak = (result.get('metrics') or {}).get('peakMemoryBytes')
    peak_text = f'{peak / 1e9:.2f}' if peak else '—'
    state = detail.get('errorCode') or detail.get('state') or 'unknown'
    lines.append(f'| [{path.stem}](results/{path.name}) | {state} | {score_text} | {rates} | {peak_text} |')
(root / 'BENCHMARK_HISTORY.md').write_text('\n'.join(lines) + '\n')
