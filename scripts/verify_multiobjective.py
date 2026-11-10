import json
from importlib.metadata import version
from pathlib import Path
import platform
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from t2d_aaf.research.crossfit import digest
from t2d_aaf.research.experiment import read_json, write_json


def verify():
    output = ROOT / 'results/multiobjective_v2'
    first = read_json(ROOT / 'results/acquisition_v2/protocol.json')
    second = read_json(output / 'protocol.json')
    signatures = [first['inputs'], first['source_sha256'], second['protected_artifacts'],
                  second['source_sha256'], joblib.load(ROOT / 'models/research/multiobjective_v2/training_signature.joblib')]
    checked = {}
    for signature in signatures:
        for name, expected in signature.items():
            actual = digest(ROOT / name)
            if actual != expected:
                raise AssertionError(f'Artifact changed: {name}')
            checked[name] = actual
    selection = read_json(output / 'integrity.json')['selection_sha256']
    if digest(output / 'selection.json') != selection:
        raise AssertionError('Frozen validation selection changed.')
    models = [ROOT / 'models/research/multiobjective_v2/labs.joblib',
              ROOT / 'models/research/acquisition_v2/oof_revision.joblib']
    summary = {'checked_artifacts': len(checked), 'all_hashes_match': True,
               'model_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in models},
               'selection_sha256': selection,
               'python': platform.python_version(),
               'packages': {name: version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'joblib', 'streamlit']},
               'commands': ['python scripts/train_multiobjective.py', 'python scripts/run_multiobjective.py',
                            'python -m pytest -q', 'python scripts/verify_multiobjective.py',
                            'streamlit run app/streamlit_app.py'],
               'clinical_sources_checked_on': '2026-11-10'}
    write_json(output / 'verification.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    verify()
