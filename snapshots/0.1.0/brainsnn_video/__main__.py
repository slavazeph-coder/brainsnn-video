"""Small local CLI. CPU commands never create a remote job or controller."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path(__file__).resolve().parent / "engine"
OBSERVATION = ENGINE / "observation_memory_v2"


def _run(arguments, directory, timeout):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-B", *arguments], cwd=directory,
                          env=env, timeout=timeout, check=False).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local BrainSNN Video research; CPU defaults, bounded evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="print snapshot identity without importing torch")
    tests = sub.add_parser("test", help="run observation layer contract checks on CPU")
    tests.add_argument("--native", action="store_true", help="also run preserved native correctness tests")
    tests.add_argument("--timeout", type=int, default=180)
    sub.add_parser("benchmark", help="run the synthetic reference benchmark; requires --out")
    sub.add_parser("train-smoke", help="run a one-update CPU integration smoke; requires --out")
    native = sub.add_parser("native", help="run a preserved native diagnostic on CPU")
    native.add_argument("tool", choices=["long_horizon", "recall_probe", "bench_scaling_honest"])
    args, remaining = parser.parse_known_args(argv)
    if args.command == "info":
        if remaining:
            parser.error("info accepts no additional arguments")
        manifest = json.loads((Path(__file__).resolve().parent / "snapshot.json").read_text())
        print(json.dumps({"package": "brainsnn-video", "version": "0.1.0",
                          "snapshot_id": manifest["snapshot_id"],
                          "source_files": len(manifest["files"]),
                          "weights_included": False, "media_included": False,
                          "gpu_jobs_started": False,
                          "evidence_scope": "CPU engineering/synthetic reference retention; learned video unproven"}, indent=2))
        return 0
    if args.command == "test":
        if remaining or args.timeout < 1:
            parser.error("test accepts --native and a positive --timeout only")
        rc = _run(["-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-v"], OBSERVATION, args.timeout)
        if rc or not args.native:
            return rc
        rc = _run(["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], ENGINE, args.timeout)
        if rc:
            return rc
        return _run(["-m", "pytest", "tests/test_rollout_training.py", "-q"], ENGINE, args.timeout)
    # Keep new outputs relative to the user's invocation directory, not package files.
    remaining = list(remaining)
    for index, argument in enumerate(remaining):
        if argument == "--out" and index + 1 < len(remaining):
            remaining[index + 1] = str(Path(remaining[index + 1]).resolve())
        elif argument.startswith("--out="):
            remaining[index] = "--out=" + str(Path(argument[6:]).resolve())
    if args.command == "benchmark":
        return _run(["run.py", *remaining], OBSERVATION, 300)
    if args.command == "train-smoke":
        return _run(["train.py", "--smoke", *remaining], OBSERVATION, 300)
    return _run([args.tool + ".py", *remaining], ENGINE, 300)


if __name__ == "__main__":
    raise SystemExit(main())
