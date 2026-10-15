"""Readable entry point for DASR experiment stages.

Keep this file beside the dasr/ directory. The model/runtime stays in separate
modules so that each experiment factor can be inspected independently.

Examples:
    python run_experiments.py check
    python run_experiments.py codecs --backend toy --split 2 --output results/debug
    python run_experiments.py codecs --backend llada --checkpoint /path/to/model \
        --revision COMMIT --split 12 --data data/eval.jsonl --output results/codecs
    python run_experiments.py analyze --input results/codecs

Run a stage with --help to see its supported arguments. No stage automatically
downloads or runs a large model unless you select a real-model experiment.
"""
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent

# These are shortcuts to the existing CLI, not new implementations of methods.
# Recipe settings are appended last so codecs/FFN/drift retain their intended scope.
RECIPES = {
    "prepare-data": ("prepare-gsm8k", []),
    "validate": ("validate-model", []),
    "bench": ("bench", []),
    "cloud": ("bench", ["--split", "0", "--codec", "full"]),
    "codecs": ("bench", ["--sweep", str(PROJECT / "dasr/configs/colab_pilot.json")]),
    "ffn": ("bench", ["--sweep", str(PROJECT / "dasr/configs/ffn_ablation.json")]),
    "drift": ("bench", ["--codec", "full", "--trace-drift"]),
    "adaptive": ("bench", []),
    "analyze": ("analyze", []),
    "profile": ("profile-table", []),
    "replay": ("replay-scheduler", []),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("Stages: check, " + ", ".join(RECIPES))
        print("Read dasr/EXPERIMENT_GUIDE.md for baselines, RQs and evaluation-table fields.")
        return 0
    stage, options = argv[0], argv[1:]
    if stage == "check":
        command = [sys.executable, "-m", "unittest", "discover", "-s", "dasr/tests", "-v"] + options
    else:
        if stage not in RECIPES:
            raise SystemExit("Unknown stage: " + stage)
        if stage == "adaptive" and "--adaptive-profile" not in options and not any(o.startswith("--adaptive-profile=") for o in options):
            if "--help" not in options and "-h" not in options:
                raise SystemExit("adaptive needs --adaptive-profile pointing to a measured calibration table")
        command_name, recipe_options = RECIPES[stage]
        command = [sys.executable, "-m", "dasr", command_name] + options + recipe_options
    # A fresh process releases model weights between stages. Working-directory
    # selection makes package imports independent of where the caller launched it.
    print("Running stage:", stage, flush=True)
    return subprocess.run(command, cwd=str(PROJECT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
