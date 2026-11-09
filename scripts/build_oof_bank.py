from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from t2d_aaf.research.crossfit import crossfit
from t2d_aaf.survival.baseline import load_cohort_splits


def main():
    parser = argparse.ArgumentParser(description='Cross-fit training-only Cox subset banks.')
    parser.add_argument('--cache', type=Path, default=ROOT / 'models/research/oof')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    frame = load_cohort_splits(ROOT / 'data/processed/primary_cohort.csv').train
    crossfit(frame, cache=args.cache, folds=args.folds, workers=args.workers)


if __name__ == '__main__':
    main()
