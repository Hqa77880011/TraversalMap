"""Six-defect straight-segment map, three forward/reverse traversals, top-L=4; independent coordinate fusion, cyclic angles and fixed-map ablation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def build_steps(args):
    output = (ROOT / args.output).resolve()
    return [
        ["-m", "traversalmap", "prepare", "--count", "200", "--seed", "17", "--output", str(output / "workload.jsonl")],
        ["-m", "traversalmap", "suite", "--data", str(output / "workload.jsonl"), "--output", str(output / "comparison")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='runs/fixed-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")

    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not args.plan and output.exists():
        parser.error("Choose a fresh --output directory")
    for step in build_steps(args):
        command = [sys.executable, *step]
        if args.plan:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
