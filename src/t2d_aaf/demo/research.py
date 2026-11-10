from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from t2d_aaf.acquisition.adaptive import (
    BIT_VALUES, build_state_matrix, load_adaptive_model_bundle, load_locked_adaptive_policy,
    prepare_adaptive_data,
)
from t2d_aaf.clinical.rules import screening_triggers
from t2d_aaf.clinical.thresholds import LABS, LAB_BITS, lab_labels
from t2d_aaf.data.cohort import CORE_ACQUIRABLE_FEATURES
from t2d_aaf.external.nagala import load_nagala_cohort
from t2d_aaf.multiobjective.policy import lab_probabilities
from t2d_aaf.research.experiment import read_json
from t2d_aaf.research.policy import PANEL_MASKS, predict_gain, MarginalCosts
from t2d_aaf.survival.baseline import PREDICTOR_FEATURES, load_cohort_splits


@dataclass
class ResearchService:
    bundle: object
    utility: dict
    labs: object
    selection: dict
    multi_selection: dict
    v1_penalty: float

    @classmethod
    def from_project(cls, root: Path):
        return cls(
            load_adaptive_model_bundle(root / 'reports/static_baselines_metrics.json', root / 'models/static_subsets.joblib'),
            {name: joblib.load(root / f'models/research/acquisition_v2/{name}.joblib')
             for name in ('insample_revision', 'oof_revision', 'oof_brier')},
            joblib.load(root / 'models/research/multiobjective_v2/labs.joblib'),
            read_json(root / 'results/acquisition_v2/selection.json'),
            read_json(root / 'results/multiobjective_v2/selection.json'),
            load_locked_adaptive_policy(root / 'models/adaptive_value_policy.joblib').selected_cost_penalty,
        )

    def prepare(self, frame):
        if tuple(frame.columns) != PREDICTOR_FEATURES:
            raise ValueError('Only the fixed predictor columns may enter the demo.')
        return prepare_adaptive_data(self.bundle, frame)

    def next_action(self, prepared, mask, criterion, costs, *, available=63, k=3, panel_start=False,
                    penalty=None, weight=None, budget=None):
        state = build_state_matrix(prepared, np.array([mask], dtype=int))
        name = 'oof_revision' if criterion == 'multiobjective' else criterion
        if criterion == 'v1':
            name = 'insample_revision'
        model = self.utility[name]
        gain = predict_gain(np.array([m.predict(state)[0] for m in model.steps]), model.criterion)
        delta = costs.delta(mask, BIT_VALUES)
        if criterion == 'multiobjective':
            weight = self.multi_selection['budgets'][str(k)]['adaptive']['weight'] if weight is None else weight
            conditional, observed = lab_probabilities(prepared, np.array([mask]), self.labs)
            discovery = np.zeros(6)
            for i, bit in enumerate(LAB_BITS):
                discovery[int(bit).bit_length() - 1] = conditional[0, i] * observed[0, i]
            score = (gain / self.multi_selection['scale'] + weight * discovery) / np.maximum(delta, 1e-12)
            budget = min(k, available.bit_count()) + costs.draw if budget is None else budget
            affordable = costs.total(mask | BIT_VALUES) <= budget + 1e-9
        else:
            if penalty is None:
                if criterion == 'v1':
                    penalty = self.v1_penalty
                else:
                    rule = 'panel_start' if panel_start else 'myopic'
                    selected = self.selection[f'draw_{costs.draw:g}']['policies'][criterion + '__' + rule]['parameter']
                    penalty = np.inf if selected is None else selected
            score = gain - penalty * delta
            affordable = np.ones(6, bool) if budget is None else costs.total(mask | BIT_VALUES) <= budget + 1e-9
        allowed = ((mask & BIT_VALUES) == 0) & ((available & BIT_VALUES) != 0) & affordable
        score[~allowed] = -np.inf
        choice = int(np.argmax(score))
        addition, best = int(BIT_VALUES[choice]), float(score[choice])
        if criterion != 'multiobjective' and panel_start and mask == 0:
            panel_gain = predict_gain(model.panels.predict(state)[0], model.criterion)
            panel_score = panel_gain - penalty * costs.total(PANEL_MASKS)
            panel_score[(PANEL_MASKS & available) != PANEL_MASKS] = -np.inf
            if budget is not None:
                panel_score[costs.total(PANEL_MASKS) > budget + 1e-9] = -np.inf
            index = int(np.argmax(panel_score))
            if panel_score[index] > best:
                addition, best = int(PANEL_MASKS[index]), float(panel_score[index])
        if not np.isfinite(best) or (criterion != 'multiobjective' and best <= 0):
            addition = 0
        return addition, [{'feature': feature, 'predicted_gain': float(gain[i]),
                           'marginal_cost': float(delta[i]),
                           'decision_score': float(score[i]) if np.isfinite(score[i]) else None,
                           'eligible': bool(allowed[i])}
                          for i, feature in enumerate(CORE_ACQUIRABLE_FEATURES)]

    def replay(self, frame, criterion, draw=0, *, available=63, k=3, panel_start=False, guideline=None):
        prepared = self.prepare(frame)
        costs = MarginalCosts(draw=draw)
        masks = [0]
        actions = ['Base']
        for _ in range(6):
            addition, _ = self.next_action(prepared, masks[-1], criterion, costs, available=available,
                                           k=k, panel_start=panel_start)
            if not addition:
                break
            masks.append(masks[-1] | addition)
            actions.append(' + '.join(feature for i, feature in enumerate(CORE_ACQUIRABLE_FEATURES) if addition & (1 << i)))
        if guideline and screening_triggers(frame, guideline, asian_ancestry=True)['require_fpg'][0] and not masks[-1] & 1:
            masks.append(masks[-1] | 1)
            actions.append('FPG: ' + guideline)
        truth = lab_labels(frame, available)[0]
        trajectory = []
        for mask, action in zip(masks, actions):
            trajectory.append({'action': action, 'mask': mask, 'risk_3y': float(prepared.subset_risks[0, mask]),
                               'cost': float(costs.total(mask)), 'test_count': mask.bit_count(),
                               'observed_abnormalities_found': int(np.sum((truth == 1) & ((mask & LAB_BITS) != 0)))})
        return trajectory, prepared


def cohort_predictors(root: Path, cohort):
    if cohort == 'internal':
        frame = load_cohort_splits(root / 'data/processed/primary_cohort.csv').test
    elif cohort == 'external':
        frame = load_nagala_cohort(root / 'data/external/Data set Int J Obesity.xlsx').mapped
    else:
        raise ValueError('Unknown cohort.')
    return frame.loc[:, PREDICTOR_FEATURES].reset_index(drop=True)
