from pathlib import Path

from client import Dryft
from package import package

LATENCY_GATE = 1.10

ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"


def report(detail: dict) -> bool:
    state = detail.get("state")
    result = detail.get("result") or {}
    shapes = result.get("shapes") or []

    print(f"run {detail.get('id')}: {state}")
    if result.get("score") is not None:
        print(f"score {result['score']:.1f} tokens/s (hidden-workload geometric mean)")

    for shape in shapes:
        metrics = shape.get("modelMetrics") or {}
        columns = [f"{shape['id']:<10}", f"{shape.get('caseStatus', '?'):<13}"]
        if shape.get("metricMs") and metrics.get("referenceMs"):
            speedup = metrics["referenceMs"] / shape["metricMs"]
            columns.append(f"{shape['metricMs']:8.1f} ms  {speedup:5.2f}x native")
        for name, mine, native in (
            ("ttft", metrics.get("ttftMs"), metrics.get("referenceTtftMs")),
            ("tpot", metrics.get("tpotMs"), metrics.get("referenceTpotMs")),
        ):
            if mine and native:
                ratio = mine / native
                flag = "  OVER GATE" if ratio > LATENCY_GATE else ""
                columns.append(f"{name} {ratio:4.2f}x{flag}")
        if shape.get("tokensPerSecond"):
            columns.append(f"{shape['tokensPerSecond']:7.1f} tok/s")
        print("  " + "  ".join(columns))
        if shape.get("caseMessage"):
            print(f"    {shape['caseMessage']}")

    for label, value in (
        ("failure", result.get("failureMessage") or result.get("failureCode")),
        ("error", detail.get("errorMessage") or detail.get("errorCode")),
        ("not ranked", result.get("rankingReason")),
    ):
        if value:
            print(f"  {label}: {value}")

    return state == "succeeded" and all(
        shape.get("caseStatus") != "failed" for shape in shapes
    )


def attempt(client: Dryft, engine_dir: Path, mode: str, timeout: float) -> bool:
    archive = package(engine_dir)
    print(f"packaged {engine_dir} -> {len(archive)} bytes")

    submission_id = client.submit(archive)
    started = client.start_run(submission_id, mode=mode)
    run_id = started["id"]
    print(f"submission {submission_id}, {mode} run {run_id}; waiting")

    return report(client.wait(run_id, timeout=timeout))


def plan_next_edit(history: list[dict]) -> str:
    raise NotImplementedError("this is the part you write")


if __name__ == "__main__":
    # Read-only monitor for the current repository-triggered official workflow.
    import argparse
    import json
    import time
    import urllib.error
    from client import ApiError, TERMINAL

    parser = argparse.ArgumentParser(description="Watch and record an existing Dryft run")
    parser.add_argument("run_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    client = Dryft()
    deadline = time.monotonic() + args.timeout
    previous_state = None
    while True:
        try:
            detail = client.run(args.run_id)
        except (urllib.error.URLError, TimeoutError, ApiError) as error:
            if isinstance(error, ApiError) and error.status not in (429, 500, 502, 503, 504):
                raise
            if time.monotonic() >= deadline:
                raise SystemExit("Polling deadline reached during network errors; run status unknown")
            print("Transient API read failure; retrying the same run in 15 seconds", flush=True)
            time.sleep(15)
            continue
        if detail.get("state") != previous_state:
            previous_state = detail.get("state")
            print(f"run {args.run_id}: {previous_state}", flush=True)
        if previous_state in TERMINAL:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(detail, indent=2) + "\n")
            passed = report(detail)
            print(f"Saved {args.output}", flush=True)
            raise SystemExit(0 if passed else 1)
        if time.monotonic() >= deadline:
            raise SystemExit("Polling deadline reached; the run remains active")
        time.sleep(15)
