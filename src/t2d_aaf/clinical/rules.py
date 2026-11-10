from __future__ import annotations

import numpy as np


SOURCES = {
    'ADA2026': 'https://doi.org/10.2337/dc26-s002',
    'KZ158': 'https://nrchd.kz/storage/documents/Клинические%20протоколы/Поток%20клинические%20протоколы/Сахарный%20диабет%202%20типа.pdf',
}


def screening_triggers(frame, guideline='ADA2026', *, asian_ancestry=False,
                       first_degree_family=None, antihypertensive=None, high_risk_ethnicity=None):
    if guideline not in SOURCES:
        raise ValueError('Unknown guideline.')
    n = len(frame)
    age = frame.age.to_numpy(float)
    bmi = frame.bmi.to_numpy(float)
    sbp, dbp = frame.sbp.to_numpy(float), frame.dbp.to_numpy(float)
    asian = np.broadcast_to(np.asarray(asian_ancestry, dtype=bool), (n,))
    family = np.zeros(n, bool) if first_degree_family is None else np.asarray(first_degree_family) == 1
    therapy = np.zeros(n, bool) if antihypertensive is None else np.asarray(antihypertensive) == 1
    if guideline == 'ADA2026':
        by_age = age >= 35
        overweight = bmi >= np.where(asian, 23., 25.)
        hypertension = (sbp >= 130) | (dbp >= 80) | therapy
        ethnic_risk = asian
    else:
        by_age = age > 45
        overweight = bmi >= 25
        hypertension = (sbp >= 140) | (dbp >= 90) | therapy
        ethnic_risk = np.zeros(n, bool) if high_risk_ethnicity is None else np.asarray(high_risk_ethnicity) == 1
    reasons = {'age': by_age, 'bmi_and_high_risk_ethnicity': overweight & ethnic_risk,
               'bmi_and_hypertension': overweight & hypertension,
               'bmi_and_first_degree_family': overweight & family}
    required = np.logical_or.reduce(list(reasons.values()))
    return {'require_fpg': required, 'reasons': reasons, 'source': SOURCES[guideline],
            'scope': 'Subset of screening indications observable before blood acquisition; not a diagnostic rule',
            'unassessed': ['CVD', 'PCOS', 'gestational diabetes', 'physical inactivity',
                           'prior dysglycemia', 'HDL/TG risk factors before acquisition'] +
                          (['first-degree family history'] if first_degree_family is None else []) +
                          (['high-risk ethnicity classification (not specified by KZ158)']
                           if guideline == 'KZ158' and high_risk_ethnicity is None else []) +
                          (['antihypertensive therapy'] if antihypertensive is None else [])}


def apply_guardrail(masks, triggers):
    masks = np.asarray(masks, dtype=int)
    return masks | np.asarray(triggers['require_fpg'], dtype=int)
