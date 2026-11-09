from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from t2d_aaf.research.experiment import run_limited


def main():
    parser = argparse.ArgumentParser(description='Compare cross-fitted acquisition utilities and marginal costs.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/acquisition_v2')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--bootstrap-workers', type=int, default=4)
    args = parser.parse_args()
    run_limited(ROOT, args.output_dir, workers=args.workers, bootstrap_workers=args.bootstrap_workers)


if __name__ == '__main__':
    main()
