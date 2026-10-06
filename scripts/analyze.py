"""Regenerate the local visualization report for an existing run."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.visualization import analyze_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a Crafter experiment without calling the model")
    parser.add_argument("run_dir", type=Path, help="One experiment directory under outputs/results/")
    args = parser.parse_args()
    print(analyze_run(args.run_dir))
