from pathlib import Path

import joblib
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from t2d_aaf.demo.research import ResearchService, cohort_predictors
from t2d_aaf.multiobjective.policy import combined_table, budget_policy
from t2d_aaf.research.policy import MarginalCosts, score_table, simulate
from t2d_aaf.triage_experiment import protected_hashes
from t2d_aaf.research.experiment import read_json


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def service():
    if not (ROOT / 'models/research/multiobjective_v2/labs.joblib').exists():
        pytest.skip('Train local multiobjective artifacts first.')
    return ResearchService.from_project(ROOT)


def test_replay_matches_frozen_experiment_policies(service):
    with threadpool_limits(limits=1):
        for cohort, available in [('internal', 63), ('external', 15)]:
            frame = cohort_predictors(ROOT, cohort).iloc[[0]]
            prepared = service.prepare(frame)
            for draw in (0, 1, 5, 20):
                costs = MarginalCosts(draw=draw)
                for criterion in ('insample_revision', 'oof_revision', 'oof_brier'):
                    table = score_table(prepared, service.utility[criterion])
                    for panel in (False, True):
                        rule = 'panel_start' if panel else 'myopic'
                        p = service.selection[f'draw_{draw}']['policies'][criterion + '__' + rule]['parameter']
                        expected = simulate(table, costs, np.inf if p is None else p, panels=panel, available=available)
                        trace, _ = service.replay(frame, criterion, draw, available=available, panel_start=panel)
                        assert trace[-1]['mask'] == expected.masks[0]
                        assert trace[-1]['cost'] == costs.total(expected.masks[0])
            revision, discovery = combined_table(prepared, service.utility['oof_revision'], service.labs, service.multi_selection['scale'])
            for k in (1, 2, 3, 4, 6):
                weight = service.multi_selection['budgets'][str(k)]['adaptive']['weight']
                expected = budget_policy(revision, discovery, weight, min(k, available.bit_count()), MarginalCosts(), available=available)
                trace, _ = service.replay(frame, 'multiobjective', available=available, k=k)
                assert trace[-1]['mask'] == expected[0]


def test_sandbox_hard_budget_and_predictor_contract(service):
    frame = cohort_predictors(ROOT, 'internal').iloc[[1]]
    with pytest.raises(ValueError, match='predictor'):
        service.prepare(frame.assign(event=1))
    prepared = service.prepare(frame)
    for criterion in ('multiobjective', 'oof_revision', 'oof_brier'):
        addition, _ = service.next_action(prepared, 0, criterion, MarginalCosts(draw=20), budget=20, penalty=0.)
        assert addition == 0


def test_training_folds_and_protected_artifacts(service):
    audit = joblib.load(ROOT / 'models/research/multiobjective_v2/audit.joblib')
    assert len(audit['folds']) == 5
    assert sum(fold['held_count'] for fold in audit['folds']) == 148283
    assert all(fold['participant_overlap'] == 0 for fold in audit['folds'])
    protocol = read_json(ROOT / 'results/multiobjective_v2/protocol.json')
    assert protected_hashes(ROOT) == protocol['protected_artifacts']
    for name in ('internal', 'external'):
        result = read_json(ROOT / f'results/multiobjective_v2/{name}.json')
        assert result['survival_bootstrap']['replicates'] == 200
        assert result['lab_bootstrap']['methods']['full_panel']['lab_micro_recall']['point_estimate'] == 1.
        assert result['lab_bootstrap']['methods']['base_fpg']['lab_micro_recall']['point_estimate'] == 0.


def test_research_tabs_render_and_switch(service):
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()
    try:
        app = AppTest.from_file(str(ROOT / 'app/streamlit_app.py'), default_timeout=60).run()
        assert not app.exception
        assert len(app.tabs) == 2
        app.selectbox[1].select('Multi-objective').run()
        assert not app.exception
        app.selectbox[0].select('NAGALA').run()
        assert not app.exception
        app.checkbox(key='open_sandbox').check().run()
        assert not app.exception
        assert any('3-летний риск' in metric.label for metric in app.metric)
        next(item for item in app.selectbox if item.label == 'Схема стоимости').select('Справочные тарифы KZT').run()
        assert not app.exception
        next(item for item in app.selectbox if item.label == 'Критерий политики').select('OOF Brier').run()
        assert not app.exception
    finally:
        st.cache_resource.clear()
        st.cache_data.clear()
