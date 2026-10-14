"""Small local CLI. CPU commands never create a remote job or controller."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import __version__

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path(__file__).resolve().parent / "engine"
OBSERVATION = ENGINE / "observation_memory_v2"
MARY = ENGINE / "mary_validation_20261013"


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
    tests.add_argument("--analytic", action="store_true", help="also run analytic postprocess invariants")
    tests.add_argument("--mary", action="store_true", help="also run local-artifact validation contracts")
    tests.add_argument("--timeout", type=int, default=180)
    sub.add_parser("benchmark", help="run the synthetic reference benchmark; requires --out")
    sub.add_parser("train-smoke", help="run a one-update CPU integration smoke; requires --out")
    sub.add_parser("preflight-smoke", help="run synthetic validator contracts; requires --out, no pretrained evidence")
    sub.add_parser("assess-native", help="frozen CPU native clip assessor; requires --input, --out and --source-role")
    native = sub.add_parser("native", help="run a preserved native diagnostic on CPU")
    native.add_argument("tool", choices=["long_horizon", "recall_probe", "bench_scaling_honest"])
    args, remaining = parser.parse_known_args(argv)
    if args.command == "info":
        if remaining:
            parser.error("info accepts no additional arguments")
        manifest = json.loads((Path(__file__).resolve().parent / "snapshot.json").read_text())
        print(json.dumps({"package": "brainsnn-video", "version": __version__,
                          "snapshot_id": manifest["snapshot_id"],
                          "source_files": len(manifest["files"]),
                          "weights_included": False, "media_included": False,
                          "gpu_jobs_started": False,
                          "known_analytic_test_failures": 3,
                          "evidence_scope": "CPU engineering/reference retention and partial cached LTX encoder check; learned video and product quality unproven"}, indent=2))
        return 0
    if args.command == "test":
        if remaining or args.timeout < 1:
            parser.error("test accepts --native, --analytic, --mary and a positive --timeout only")
        rc = _run(["-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-v"], OBSERVATION, args.timeout)
        if rc:
            return rc
        if args.native:
            rc = _run(["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], ENGINE, args.timeout)
            if not rc:
                rc = _run(["-m", "pytest", "tests/test_rollout_training.py", "-q"], ENGINE, args.timeout)
            if rc:
                return rc
        if args.mary:
            rc = _run(["-m", "unittest", "test_validation", "-v"], MARY, args.timeout)
            if rc:
                return rc
        if args.analytic:
            return _run(["-m", "pytest", "crumb_coherence/tests/test_core.py", "-q"], ENGINE, args.timeout)
        return 0
    # Keep new outputs relative to the user's invocation directory, not package files.
    remaining = list(remaining)
    for index, argument in enumerate(remaining):
        if argument in ("--out", "--input") and index + 1 < len(remaining):
            remaining[index + 1] = str(Path(remaining[index + 1]).resolve())
        elif argument.startswith("--out="):
            remaining[index] = "--out=" + str(Path(argument[6:]).resolve())
        elif argument.startswith("--input="):
            remaining[index] = "--input=" + str(Path(argument[8:]).resolve())
    if args.command == "benchmark":
        return _run(["run.py", *remaining], OBSERVATION, 300)
    if args.command == "train-smoke":
        return _run(["train.py", "--smoke", *remaining], OBSERVATION, 300)
    if args.command == "preflight-smoke":
        return _run(["run_preflight.py", "--smoke", *remaining], MARY, 180)
    if args.command == "assess-native":
        return _run([str(Path(__file__).resolve().parent / "assess_native.py"), *remaining], ROOT, 1900)
    return _run([args.tool + ".py", *remaining], ENGINE, 300)


if __name__ == "__main__":
    raise SystemExit(main())
