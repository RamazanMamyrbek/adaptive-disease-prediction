from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from t2d_aaf.acquisition.adaptive import load_adaptive_model_bundle, prepare_adaptive_data, subset_to_mask
from t2d_aaf.acquisition.ordered import build_train_threshold_candidates, simulate_ordered_policy
from t2d_aaf.evaluation.survival_metrics import harrell_c_index
from t2d_aaf.external.nagala import COMPATIBLE_ACQUIRABLE_FEATURES, load_nagala_cohort
from t2d_aaf.external.validation import load_adaptive_value_models
from t2d_aaf.research.crossfit import atomic_dump, crossfit, digest
from t2d_aaf.research.inference import paired_intervals
from t2d_aaf.research.policy import MarginalCosts, fit_utilities, penalty_grid, score_table, simulate
from t2d_aaf.survival.baseline import load_cohort_splits


CRITERIA = ('insample_revision', 'oof_revision', 'oof_brier')
RULES = ('myopic', 'panel_start')
BASELINES = ('base_only', 'base_fpg', 'static_greedy', 'ordered', 'full_panel')
COUNTS = np.array([mask.bit_count() for mask in range(64)])


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def number(value):
    return None if np.isinf(value) else float(value)


def decode(value):
    return np.inf if value is None else float(value)


def choose_point(curve, target):
    feasible = [point for point in curve if point['validation_c_index'] >= target]
    if feasible:
        selected = min(feasible, key=lambda p: (p['mean_cost'], -p['validation_c_index']))
    else:
        selected = min(curve, key=lambda p: (-p['validation_c_index'], p['mean_cost']))
    return {**selected, 'retention_constraint_met': bool(feasible), 'target_validation_c_index': target}


def evaluate_point(prepared, frame, masks, costs, parameter):
    risk = prepared.subset_risks[np.arange(len(masks)), masks]
    return {'parameter': number(parameter), 'validation_c_index': harrell_c_index(frame.duration_years, frame.event, risk),
            'mean_cost': float(costs.total(masks).mean()), 'mean_tests': float(COUNTS[masks].mean()),
            'stop_without_tests_fraction': float(np.mean(masks == 0))}


def prefix_masks(order, available=63):
    values = [0]
    for name in order:
        bit = subset_to_mask((name,))
        if bit & available:
            values.append(values[-1] | bit)
    return np.asarray(values, dtype=int)


def ordered_masks(prepared, order, threshold, available=63):
    masks = prefix_masks(order, available)
    policy = simulate_ordered_policy(prepared.subset_risks[:, masks], threshold)
    return masks[policy.stop_stage]


def baseline_masks(prepared, order, selected, available=63):
    count = prepared.participants
    return {
        'base_only': np.zeros(count, dtype=int),
        'base_fpg': np.full(count, subset_to_mask(('fpg',)) & available, dtype=int),
        'full_panel': np.full(count, available, dtype=int),
        'static_greedy': np.full(count, int(selected['static_greedy']['parameter']) & available, dtype=int),
        'ordered': ordered_masks(prepared, order, decode(selected['ordered']['parameter']), available),
    }


def baseline_selection(train, validation, frame, order, costs, target):
    prefixes = prefix_masks(order)
    static_curve = [evaluate_point(validation, frame, np.full(len(frame), mask, dtype=int), costs, mask)
                    for mask in prefixes]
    thresholds = build_train_threshold_candidates(train.subset_risks[:, prefixes])
    ordered_curve = [evaluate_point(validation, frame, ordered_masks(validation, order, threshold), costs, threshold)
                     for threshold in thresholds]
    return {'static_greedy': choose_point(static_curve, target), 'ordered': choose_point(ordered_curve, target)}, {
        'static_greedy': static_curve, 'ordered': ordered_curve}


def study_inputs(root):
    paths = ['data/processed/primary_cohort.csv', 'data/processed/primary_cohort_manifest.json',
             'data/external/Data set Int J Obesity.xlsx', 'models/static_subsets.joblib',
             'models/adaptive_value_policy.joblib', 'models/cox_baseline.joblib']
    for directory in ('reports', 'configs'):
        paths.extend(path.relative_to(root).as_posix() for path in (root / directory).glob('*') if path.is_file())
    return {name: digest(root / name) for name in sorted(set(paths))}


def export_tables(output, cohorts):
    for filename, collection in (('metrics.csv', 'methods'), ('contrasts.csv', 'contrasts')):
        with (output / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['cohort', 'method', 'metric', 'point_estimate',
                                                        'ci95_lower', 'ci95_upper', 'valid_replicates'])
            writer.writeheader()
            for cohort, result in cohorts.items():
                for name, values in result['bootstrap'][collection].items():
                    for metric, interval in values.items():
                        writer.writerow({'cohort': cohort, 'method': name, 'metric': metric, **interval})


def run(root: Path, output: Path, *, workers=3, bootstrap_workers=4):
    started = time.monotonic()
    config = read_json(root / 'configs/acquisition_research.json')
    output = output.resolve()
    if output == root or not output.is_relative_to(root / 'results'):
        raise ValueError('Aggregate results must be in a dedicated results subdirectory.')
    if (output / 'metrics.json').exists():
        raise ValueError('Completed results are immutable; choose a new output directory.')
    output.mkdir(parents=True, exist_ok=True)
    inputs = study_inputs(root)
    source_paths = sorted((root / 'src').rglob('*.py')) + [root / 'scripts/run_acquisition_research.py']
    sources = {path.relative_to(root).as_posix(): digest(path) for path in source_paths}
    identity = {'protocol': config, 'inputs': inputs, 'source_sha256': sources}
    identity_path = output / 'protocol.json'
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError('Experiment inputs or implementation changed; use a new output directory.')
    write_json(identity_path, identity)
    cache = root / 'models/research' / output.name
    cache.mkdir(parents=True, exist_ok=True)
    cache_signature = cache / 'signature.joblib'
    if cache_signature.exists() and joblib.load(cache_signature) != identity:
        raise ValueError('Local checkpoint identity mismatch.')
    atomic_dump(identity, cache_signature)
    splits = load_cohort_splits(root / 'data/processed/primary_cohort.csv')
    oof = crossfit(splits.train, cache=root / 'models/research/oof', folds=config['folds'],
                   seed=config['fold_seed'], horizon=config['horizon_years'], workers=workers)
    write_json(output / 'crossfit_audit.json', {'folds': oof['audit'], 'signature': oof['signature']})
    bundle = load_adaptive_model_bundle(root / 'reports/static_baselines_metrics.json', root / 'models/static_subsets.joblib')
    train = prepare_adaptive_data(bundle, splits.train)
    validation = prepare_adaptive_data(bundle, splits.validation)
    order = tuple(bundle.static_results['methods']['importance_greedy']['order'])
    base_c = harrell_c_index(splits.validation.duration_years, splits.validation.event, validation.subset_risks[:, 0])
    full_c = harrell_c_index(splits.validation.duration_years, splits.validation.event, validation.subset_risks[:, 63])
    if full_c <= base_c:
        raise ValueError('Retention criterion requires full model improvement over base.')
    target = base_c + config['retention_fraction'] * (full_c - base_c)
    models, scores = {}, {}
    for name in CRITERIA:
        path = cache / (name + '.joblib')
        if path.exists():
            model = joblib.load(path)
        else:
            prepared = train if name == 'insample_revision' else oof['prepared']
            model = fit_utilities(prepared, 'brier' if name == 'oof_brier' else 'revision',
                                  labels=oof['labels'], weights=oof['weights'],
                                  seed=config['state_seed'], alpha=config['ridge_alpha'])
            atomic_dump(model, path)
        models[name] = model
        scores[name] = score_table(validation, model)
        print(f'{name}: value regressors ready', flush=True)
    original = load_adaptive_value_models(root / 'models/adaptive_value_policy.joblib')
    for actual, expected in zip(models['insample_revision'].steps, original.models):
        np.testing.assert_allclose(actual.coef_, expected.estimator.coef_, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(actual.intercept_, expected.estimator.intercept_, rtol=1e-10, atol=1e-12)
    write_json(output / 'utility_training.json', {name: model.training_summary for name, model in models.items()})
    selection_path = output / 'selection.json'
    selections = read_json(selection_path) if selection_path.exists() else {}
    for draw in config['draw_costs']:
        key = f'draw_{draw:g}'
        if key in selections:
            continue
        print(f'Validation selection: {key}', flush=True)
        costs = MarginalCosts(tuple(config['test_costs']), draw)
        baselines, baseline_curves = baseline_selection(train, validation, splits.validation, order, costs, target)
        entry = {'baselines': baselines, 'baseline_curves': baseline_curves, 'policies': {}, 'curves': {}}
        for name in CRITERIA:
            prepared = train if name == 'insample_revision' else oof['prepared']
            for rule in RULES:
                panels = rule == 'panel_start'
                grid = penalty_grid(prepared, models[name], costs, quantiles=config['lambda_quantiles'], panels=panels)
                curve = [evaluate_point(validation, splits.validation,
                                        simulate(scores[name], costs, value, panels=panels).masks, costs, value) for value in grid]
                method = name + '__' + rule
                entry['policies'][method] = choose_point(curve, target)
                entry['curves'][method] = curve
                selected = entry['policies'][method]
                print(f"{key} {method}: C={selected['validation_c_index']:.5f}, cost={selected['mean_cost']:.3f}, feasible={selected['retention_constraint_met']}", flush=True)
        selections[key] = entry
        write_json(selection_path, selections)
    del scores, validation, train, oof
    selection_hash = digest(selection_path)
    cohorts = {}
    for cohort_name in ('internal', 'external'):
        completed = output / (cohort_name + '.json')
        if completed.exists():
            cohorts[cohort_name] = read_json(completed)
            continue
        print(f'{cohort_name}: evaluating frozen policies', flush=True)
        if cohort_name == 'internal':
            frame, censoring, available = splits.test, splits.train, 63
        else:
            frame = load_nagala_cohort(root / 'data/external/Data set Int J Obesity.xlsx').mapped
            censoring, available = frame, subset_to_mask(COMPATIBLE_ACQUIRABLE_FEATURES)
        prepared = prepare_adaptive_data(bundle, frame)
        rows = np.arange(len(frame))
        risks, costs_by_method, counts, metadata, contrasts = {}, {}, {}, {}, []

        def record(name, masks, costs, info):
            risks[name] = prepared.subset_risks[rows, masks]
            costs_by_method[name] = costs.total(masks)
            counts[name] = COUNTS[masks]
            metadata[name] = info

        for draw in config['draw_costs']:
            key = f'draw_{draw:g}'
            costs = MarginalCosts(tuple(config['test_costs']), draw)
            for baseline, masks in baseline_masks(prepared, order, selections[key]['baselines'], available).items():
                record(baseline + '__' + key, masks, costs, {'family': 'baseline', 'draw_cost': draw})
        for name in CRITERIA:
            table = score_table(prepared, models[name])
            for draw in config['draw_costs']:
                key = f'draw_{draw:g}'
                costs = MarginalCosts(tuple(config['test_costs']), draw)
                for rule in RULES:
                    method = name + '__' + rule
                    selected = selections[key]['policies'][method]
                    run_result = simulate(table, costs, decode(selected['parameter']), panels=rule == 'panel_start', available=available)
                    label = method + '__' + key
                    record(label, run_result.masks, costs, {
                        'family': 'adaptive', 'criterion': name, 'rule': rule, 'draw_cost': draw,
                        'selected_on': 'primary validation only', 'penalty': selected['parameter'],
                        'primary_retention_constraint_met': selected['retention_constraint_met'],
                        'initial_multitest_panel_fraction': float(np.mean(COUNTS[run_result.initial_masks] > 1)),
                    })
                    contrasts.extend((label, baseline + '__' + key) for baseline in BASELINES)
                contrasts.append((name + '__panel_start__' + key, name + '__myopic__' + key))
                if draw > 0:
                    fixed = selections['draw_0']['policies'][name + '__myopic']['parameter']
                    labels = []
                    for rule in RULES:
                        run_result = simulate(table, costs, decode(fixed), panels=rule == 'panel_start', available=available)
                        label = name + '__fixed_penalty_' + rule + '__' + key
                        labels.append(label)
                        record(label, run_result.masks, costs, {
                            'family': 'fixed_penalty_diagnostic', 'criterion': name, 'rule': rule,
                            'draw_cost': draw, 'penalty': fixed,
                            'initial_multitest_panel_fraction': float(np.mean(COUNTS[run_result.initial_masks] > 1)),
                        })
                    contrasts.append((labels[1], labels[0]))
            del table
        for draw in config['draw_costs']:
            key = f'draw_{draw:g}'
            for rule in RULES:
                for name in ('oof_revision', 'oof_brier'):
                    contrasts.append((name + '__' + rule + '__' + key, 'insample_revision__' + rule + '__' + key))
                contrasts.append(('oof_brier__' + rule + '__' + key, 'oof_revision__' + rule + '__' + key))
        print(f'{cohort_name}: {len(risks)} methods, {config["bootstrap_replicates"]} paired bootstrap replicates', flush=True)
        bootstrap = paired_intervals(censoring, frame, risks, costs_by_method, counts, contrasts,
                                     horizon=config['horizon_years'], replicates=config['bootstrap_replicates'],
                                     seed=config['bootstrap_seed'], workers=bootstrap_workers)
        result = {'participants': len(frame), 'events': int(frame.event.sum()), 'available_mask': available,
                  'censoring_source': 'primary training' if cohort_name == 'internal' else 'NAGALA evaluation cohort',
                  'selection_sha256': selection_hash, 'method_metadata': metadata, 'bootstrap': bootstrap}
        write_json(completed, result)
        cohorts[cohort_name] = result
    if study_inputs(root) != inputs or digest(selection_path) != selection_hash:
        raise RuntimeError('Protected inputs or locked selection changed during the experiment.')
    if {path.relative_to(root).as_posix(): digest(path) for path in source_paths} != sources:
        raise RuntimeError('Research implementation changed during the experiment.')
    export_tables(output, cohorts)
    document = {'schema': 'acquisition_research_results_v2', 'protocol': config,
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                'packages': {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'joblib')},
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'source_sha256': sources, 'input_sha256': inputs, 'v1_inputs_unchanged': True,
                'in_sample_ridge_reproduced': True, 'test_used_for_selection': False,
                'external_used_for_selection': False, 'elapsed_seconds_this_execution': time.monotonic() - started,
                'results': cohorts}
    write_json(output / 'metrics.json', document)
    print(f'Completed: {output}', flush=True)
    return document


def run_limited(*args, **kwargs):
    with threadpool_limits(limits=1):
        return run(*args, **kwargs)
