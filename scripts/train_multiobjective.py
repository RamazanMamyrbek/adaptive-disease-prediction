from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from t2d_aaf.multiobjective.training import train
from t2d_aaf.survival.baseline import load_cohort_splits


if __name__ == '__main__':
    splits = load_cohort_splits(ROOT / 'data/processed/primary_cohort.csv')
    train(ROOT, splits.train, ROOT / 'models/research/multiobjective_v2')
