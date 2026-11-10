from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from t2d_aaf.acquisition.adaptive import load_adaptive_model_bundle, prepare_adaptive_data, build_state_matrix
from t2d_aaf.clinical.rules import SOURCES as GUIDELINES, screening_triggers, apply_guardrail
from t2d_aaf.clinical.thresholds import LABS, LAB_BITS, SOURCES as THRESHOLDS, lab_labels
from t2d_aaf.evaluation.survival_metrics import harrell_c_index
from t2d_aaf.external.nagala import load_nagala_cohort
from t2d_aaf.multiobjective.evaluation import lab_intervals
from t2d_aaf.multiobjective.policy import combined_table, budget_policy
from t2d_aaf.research.crossfit import digest
from t2d_aaf.research.experiment import read_json, write_json
from t2d_aaf.research.inference import paired_intervals
from t2d_aaf.research.policy import MarginalCosts
from t2d_aaf.survival.baseline import load_cohort_splits


BITS = (1, 2, 4, 8, 16, 32)
COUNTS = np.array([m.bit_count() for m in range(64)])


def protected_hashes(root):
    old = read_json(root / 'results/acquisition_v2/protocol.json')
    names = set(old['inputs']) | set(old['source_sha256'])
    names |= {p.relative_to(root).as_posix() for p in (root / 'results/acquisition_v2').glob('*') if p.is_file()}
    return {name: digest(root / name) for name in sorted(names)}


def recall(labels, masks):
    positive = labels == 1
    denominator = positive.sum()
    return float((positive & ((masks[:, None] & LAB_BITS) != 0)).sum() / denominator) if denominator else 0.


def point(frame, prepared, labels, masks):
    return {'c_index': harrell_c_index(frame.duration_years, frame.event,
                                      prepared.subset_risks[np.arange(len(frame)), masks]),
            'lab_recall': recall(labels, masks)}


def select(curve, target):
    feasible = [item for item in curve if item['c_index'] >= target]
    if feasible:
        best = max(feasible, key=lambda item: (item['lab_recall'], item['c_index'], -item['weight']))
    else:
        best = max(curve, key=lambda item: (item['c_index'], item['lab_recall'], -item['weight']))
    return {**best, 'retention_constraint_met': bool(feasible), 'target_c_index': target}


def static_order(frame, prepared, labels, weight, scale, *, greedy):
    base = prepared.subset_risks[:, 0]
    order, mask = [], 0
    for _ in range(6):
        current = mask if greedy else 0
        def gain(bit):
            revision = np.abs(prepared.subset_risks[:, current | bit] - prepared.subset_risks[:, current]).mean() / scale
            target = np.flatnonzero(LAB_BITS == bit)
            discovery = float(np.mean(labels[:, target[0]] == 1)) if len(target) else 0.
            return revision + weight * discovery
        bit = max((b for b in BITS if b not in order), key=lambda b: (gain(b), -b))
        order.append(bit)
        mask |= bit
    return order


def prefix(order, k, available):
    eligible = [bit for bit in order if bit & available]
    return int(sum(eligible[:k]))


def run(root: Path, output: Path, workers=6):
    output = output.resolve()
    if not output.is_relative_to(root / 'results') or output == root / 'results/acquisition_v2':
        raise ValueError('Use a separate results subdirectory.')
    if (output / 'metrics.csv').exists():
        raise ValueError('Completed experiment is immutable; choose a new output directory.')
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(root / 'configs/multiobjective_research.json')
    protected = protected_hashes(root)
    paths = list((root / 'src/t2d_aaf/multiobjective').glob('*.py'))
    paths += list((root / 'src/t2d_aaf/clinical').glob('*.py')) + [Path(__file__)]
    identity = {'config': config, 'protected_artifacts': protected,
                'source_sha256': {p.relative_to(root).as_posix(): digest(p) for p in paths},
                'thresholds': THRESHOLDS, 'guideline_sources': GUIDELINES,
                'source_check_date': '2026-11-10',
                'inference': 'Paired event-stratified percentile bootstrap; fitted models, selection and censoring weights fixed; no multiplicity correction',
                'limitations': ['Retrospective observed-positive yield depends on source missingness; unobserved truth unknown',
                               'ALT/BUN cutoffs are operational research labels, not diagnoses',
                               'Guardrails implement available screening indications, not full guideline compliance',
                               'Fixed-budget allocation differs from economic stopping in acquisition_v2',
                               'External censoring distribution is estimated from NAGALA for evaluation only']}
    protocol = output / 'protocol.json'
    if protocol.exists() and read_json(protocol) != identity:
        raise ValueError('Experiment identity changed; use a new output directory.')
    write_json(protocol, identity)
    cache = root / 'models/research/multiobjective_v2'
    labs = joblib.load(cache / 'labs.joblib')
    write_json(output / 'training_audit.json', joblib.load(cache / 'audit.joblib'))
    survival = joblib.load(root / 'models/research/acquisition_v2/oof_revision.joblib')
    bundle = load_adaptive_model_bundle(root / 'reports/static_baselines_metrics.json', root / 'models/static_subsets.joblib')
    splits = load_cohort_splits(root / 'data/processed/primary_cohort.csv')
    selection_path = output / 'selection.json'
    with threadpool_limits(limits=1):
        if selection_path.exists():
            selection = read_json(selection_path)
        else:
            oof = joblib.load(root / 'models/research/oof/oof.joblib')['prepared']
            base_state = build_state_matrix(oof, np.zeros(oof.participants, int))
            gains = np.column_stack([np.clip(m.predict(base_state), 0, 1) for m in survival.steps])
            scale = float(np.quantile(gains[gains > 0], .95))
            del oof, base_state, gains
            prepared = prepare_adaptive_data(bundle, splits.validation)
            labels = lab_labels(splits.validation)
            revision, discovery = combined_table(prepared, survival, labs, scale)
            base_c = harrell_c_index(splits.validation.duration_years, splits.validation.event, prepared.subset_risks[:, 0])
            full_c = harrell_c_index(splits.validation.duration_years, splits.validation.event, prepared.subset_risks[:, 63])
            target = base_c + config['survival_retention_fraction'] * (full_c - base_c)
            selection = {'scale': scale, 'survival_target': target, 'budgets': {},
                         'ordered_bits': [1, 2, 4, 8, 32, 16],
                         'ordered_basis': 'Prespecified FPG, TC, TG, ALT, BUN, CCR; no test-outcome tuning'}
            for k in config['test_budgets']:
                curve = []
                for weight in config['lab_weight_grid']:
                    masks = budget_policy(revision, discovery, weight, k, MarginalCosts())
                    curve.append({'weight': weight, **point(splits.validation, prepared, labels, masks)})
                best = select(curve, target)
                static = {}
                for name, greedy in [('static_topk', False), ('static_greedy', True)]:
                    candidates = []
                    for weight in config['lab_weight_grid']:
                        order = static_order(splits.validation, prepared, labels, weight, scale, greedy=greedy)
                        masks = np.full(len(labels), prefix(order, k, 63), int)
                        candidates.append({'weight': weight, 'order': order,
                                           **point(splits.validation, prepared, labels, masks)})
                    static[name] = {'selected': select(candidates, target), 'curve': candidates}
                selection['budgets'][str(k)] = {'adaptive': best, 'curve': curve, **static}
                print(f'Validation K={k}: weight={best["weight"]}, C={best["c_index"]:.6f}, recall={best["lab_recall"]:.4f}, feasible={best["retention_constraint_met"]}', flush=True)
            write_json(selection_path, selection)
            del prepared, revision, discovery
    selected_hash = digest(selection_path)
    cohorts = {}
    for name in ('internal', 'external'):
        path = output / f'{name}.json'
        if path.exists():
            cohorts[name] = read_json(path)
            continue
        print(f'{name}: evaluating frozen multiobjective policies', flush=True)
        if name == 'internal':
            frame, available, censoring = splits.test, 63, splits.train
        else:
            frame = load_nagala_cohort(root / 'data/external/Data set Int J Obesity.xlsx').mapped
            available, censoring = 15, frame
        with threadpool_limits(limits=1):
            prepared = prepare_adaptive_data(bundle, frame)
            revision, discovery = combined_table(prepared, survival, labs, selection['scale'])
            labels = lab_labels(frame, available)
            masks = {'base_only': np.zeros(len(frame), int), 'base_fpg': np.ones(len(frame), int),
                     'full_panel': np.full(len(frame), available, int)}
            contrasts, metadata = [], {}
            for k in config['test_budgets']:
                item = selection['budgets'][str(k)]
                capped = min(k, available.bit_count())
                adaptive = f'adaptive_k{k}'
                masks[adaptive] = budget_policy(revision, discovery, item['adaptive']['weight'], capped,
                                                MarginalCosts(), available=available)
                baselines = ['base_only', 'base_fpg', 'full_panel']
                for method in ('static_topk', 'static_greedy', 'ordered'):
                    order = selection['ordered_bits'] if method == 'ordered' else item[method]['selected']['order']
                    label = f'{method}_k{k}'
                    masks[label] = np.full(len(frame), prefix(order, capped, available), int)
                    baselines.append(label)
                    if not np.all(COUNTS[masks[label]] == COUNTS[masks[adaptive]]):
                        raise AssertionError('Budget-matched comparator has different test count.')
                contrasts.extend((adaptive, baseline) for baseline in baselines)
                metadata[adaptive] = {'weight': item['adaptive']['weight'], 'test_budget': capped,
                                      'validation_retention_met': item['adaptive']['retention_constraint_met']}
                for guideline in GUIDELINES:
                    for asian in (True, False):
                        triggers = screening_triggers(frame, guideline, asian_ancestry=asian)
                        label = adaptive + '__' + guideline + ('' if asian else '_ancestry_unknown')
                        masks[label] = apply_guardrail(masks[adaptive], triggers)
                        metadata[label] = {'guideline': guideline, 'asian_assumption': asian,
                                           'trigger_fraction': float(triggers['require_fpg'].mean()),
                                           'fpg_added_fraction': float(np.mean(masks[label] != masks[adaptive])),
                                           'budget_exceedance_fraction': float(np.mean(COUNTS[masks[label]] > capped)),
                                           'unassessed': triggers['unassessed']}
                        contrasts.append((label, adaptive))
                        contrasts.extend((label, baseline) for baseline in baselines)
            del revision, discovery
            risks, prices, counts, all_contrasts = {}, {}, {}, []
            for draw in config['draw_costs']:
                costs = MarginalCosts(draw=draw)
                suffix = f'__draw{draw}'
                for method, state in masks.items():
                    risks[method + suffix] = prepared.subset_risks[np.arange(len(frame)), state]
                    prices[method + suffix] = costs.total(state)
                    counts[method + suffix] = COUNTS[state]
                all_contrasts.extend((left + suffix, right + suffix) for left, right in contrasts)
            intervals = paired_intervals(censoring, frame, risks, prices, counts, all_contrasts,
                                         replicates=config['bootstrap_replicates'], seed=config['bootstrap_seed'], workers=workers)
            print(f'{name}: survival intervals complete; observed laboratory intervals', flush=True)
            lab_ci = lab_intervals(frame, labels, masks, contrasts, config['bootstrap_replicates'], config['bootstrap_seed'])
            prevalence = {lab: {'observed': int(np.isfinite(labels[:, i]).sum()),
                                'positive': int(np.nansum(labels[:, i])),
                                'unobserved': int(np.isnan(labels[:, i]).sum())} for i, lab in enumerate(LABS)}
            result = {'participants': len(frame), 'available_mask': available, 'source_labels': prevalence,
                      'metadata': metadata, 'survival_bootstrap': intervals, 'lab_bootstrap': lab_ci}
            write_json(path, result)
            cohorts[name] = result
    if digest(selection_path) != selected_hash or protected_hashes(root) != protected:
        raise AssertionError('Frozen selection or protected artifacts changed during experiment.')
    for filename, collection in [('metrics.csv', 'methods'), ('contrasts.csv', 'contrasts')]:
        with (output / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['cohort', 'method', 'metric', 'point_estimate', 'ci95_lower', 'ci95_upper', 'valid_replicates'])
            writer.writeheader()
            for cohort, result in cohorts.items():
                for family in ('survival_bootstrap', 'lab_bootstrap'):
                    for method, metrics in result[family][collection].items():
                        for metric, values in metrics.items():
                            writer.writerow({'cohort': cohort, 'method': method, 'metric': metric, **values})
    write_json(output / 'integrity.json', {'protected_artifacts_unchanged': True, 'selection_sha256': selected_hash,
                                          'cohorts': list(cohorts)})
    print(f'Completed: {output}', flush=True)
