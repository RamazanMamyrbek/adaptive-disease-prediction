import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from t2d_aaf.triage_experiment import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/multiobjective_v2')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    run(ROOT, args.output, args.workers)
