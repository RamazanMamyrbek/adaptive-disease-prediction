from __future__ import annotations

import numpy as np


LABS = ('total_cholesterol', 'triglycerides', 'alt', 'bun')
LAB_BITS = np.array([2, 4, 8, 32], dtype=int)
SOURCES = {
    'total_cholesterol': {'threshold': 5.2, 'operator': '>=', 'unit': 'mmol/L',
                          'basis': 'Prespecified research cutoff; not a treatment threshold'},
    'triglycerides': {'threshold': 1.7, 'operator': '>=', 'unit': 'mmol/L',
                      'basis': 'Prespecified research cutoff; distinct from ADA screening risk factor'},
    'alt': {'male_threshold': 33., 'female_threshold': 25., 'operator': '>', 'unit': 'U/L',
            'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10735173/',
            'basis': 'Upper bounds of AASLD healthy ranges; operational research label, not liver diagnosis'},
    'bun': {'threshold': 7.1, 'operator': '>', 'unit': 'mmol/L',
            'source': 'https://www.mayoclinic.org/tests-procedures/blood-urea-nitrogen/about/pac-20384821',
            'basis': 'Adult upper reference bound; age/laboratory reference intervals may differ'},
}


def lab_labels(frame, available=63):
    values = frame.loc[:, LABS].to_numpy(float)
    sex = frame.gender.to_numpy(float)
    threshold = np.column_stack((np.full(len(frame), 5.2), np.full(len(frame), 1.7),
                                 np.where(sex == 1, 33., 25.), np.full(len(frame), 7.1)))
    observed = np.isfinite(values) & ((LAB_BITS & available) != 0)
    observed[:, 2] &= np.isin(sex, [1., 2.])
    positive = values > threshold
    positive[:, :2] = values[:, :2] >= threshold[:, :2]
    return np.where(observed, positive.astype(float), np.nan)
