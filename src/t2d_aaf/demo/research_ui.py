from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from t2d_aaf.clinical.rules import SOURCES, screening_triggers
from t2d_aaf.clinical.thresholds import LABS, LAB_BITS, lab_labels
from t2d_aaf.data.cohort import BASE_FEATURES, CORE_ACQUIRABLE_FEATURES
from t2d_aaf.demo.research import ResearchService, cohort_predictors
from t2d_aaf.multiobjective.policy import lab_probabilities
from t2d_aaf.research.experiment import read_json
from t2d_aaf.research.policy import MarginalCosts
from t2d_aaf.survival.baseline import PREDICTOR_FEATURES


CRITERIA = {'In-sample': 'insample_revision', 'OOF Revision': 'oof_revision',
            'OOF Brier': 'oof_brier', 'Multi-objective': 'multiobjective'}


@st.cache_resource(show_spinner='Загрузка фиксированных моделей…')
def research_service(root):
    return ResearchService.from_project(root)


@st.cache_data(show_spinner='Загрузка обезличенных признаков когорты…')
def predictors(root, cohort):
    return cohort_predictors(root, cohort)


def render_probabilities(service, prepared, frame, mask, available):
    conditional, observed = lab_probabilities(prepared, np.array([mask]), service.labs)
    truth = lab_labels(frame, available)[0]
    rows = []
    for i, (lab, bit) in enumerate(zip(LABS, LAB_BITS)):
        if not bit & available:
            continue
        revealed = bool(mask & bit)
        rows.append({'Анализ': lab, 'Статус': ('Не измерено в источнике' if np.isnan(truth[i]) else
                                               'Выше порога' if truth[i] else 'Не выше порога') if revealed else 'Скрыт',
                     'Наблюдаемое значение': float(frame.iloc[0][lab]) if revealed and pd.notna(frame.iloc[0][lab]) else None,
                     'P(отклонение | значение наблюдаемо)': None if revealed else float(conditional[0, i]),
                     'P(значение наблюдаемо в источнике)': None if revealed else float(observed[0, i])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption('Пороги: TC ≥ 5.2, TG ≥ 1.7, BUN > 7.1 ммоль/л; ALT > 33 у мужчин, > 25 у женщин Ед/л. '
               'Это исследовательские лабораторные метки. Пропуски не считаются нормой или отклонением.')


def render_replay(root, cohort, criterion):
    service = research_service(root)
    frame = predictors(root, cohort)
    if st.session_state.get('replay_cohort') != cohort:
        st.session_state['replay_case'] = 1
        st.session_state['replay_step'] = 0
        st.session_state['replay_cohort'] = cohort
    a, b, c = st.columns(3)
    case = a.number_input('Номер обезличенного случая', min_value=1, max_value=len(frame), value=1, key='replay_case')
    draw = b.selectbox('Стоимость забора крови', [0, 1, 5, 20], key='replay_draw')
    k = c.selectbox('Бюджет анализов', [1, 2, 3, 4, 6], index=2, disabled=criterion != 'multiobjective')
    panel = st.checkbox('Оценивать стартовую панель', disabled=criterion in ('multiobjective', 'v1'))
    guideline = st.selectbox('Правила после статистического выбора', ['Нет', 'ADA2026', 'KZ158'])
    available = 63 if cohort == 'internal' else 15
    row = frame.iloc[[int(case) - 1]].copy()
    trajectory, prepared = service.replay(row, criterion, draw, available=available, k=k,
                                           panel_start=panel, guideline=None if guideline == 'Нет' else guideline)
    st.dataframe(row.loc[:, BASE_FEATURES], hide_index=True, width='stretch')
    if criterion == 'multiobjective':
        selected = service.multi_selection['budgets'][str(k)]['adaptive']
        st.caption(f"Вес лабораторной цели: {selected['weight']:g}; выбор на внутренней валидации. "
                   'Бюджет расходуется полностью; остановка по цене из Этапа 1 здесь не применяется.')
        if not selected['retention_constraint_met']:
            st.warning('Для этого бюджета ограничение сохранения C-index на валидации не выполнено.')
    if guideline != 'Нет':
        st.caption('Для ADA азиатское происхождение предполагается по контексту когорты; '
                   'в КП РК этнический фактор не оценивается без уточненной классификации. '
                   'FPG по правилам может увеличить исходный бюджет; это включено в стоимость.')
    table = pd.DataFrame(trajectory)
    st.dataframe(table.drop(columns='mask'), hide_index=True, width='stretch')
    st.line_chart(table[['risk_3y']], x_label='Шаг', y_label='3-летний риск')
    if st.session_state.get('replay_step', 0) >= len(trajectory):
        st.session_state['replay_step'] = 0
    step = st.select_slider('Показать состояние маршрута', options=list(range(len(trajectory))), key='replay_step')
    mask = trajectory[step]['mask']
    render_probabilities(service, prepared, row, mask, available)
    with st.expander('Результаты экспериментов и 95% интервалы'):
        path = root / 'results' / ('multiobjective_v2' if criterion == 'multiobjective' else 'acquisition_v2') / 'metrics.csv'
        if path.exists():
            data = pd.read_csv(path)
            st.dataframe(data[data.cohort == cohort], hide_index=True, width='stretch')
            st.download_button('Скачать таблицу метрик', path.read_bytes(), file_name=path.name, mime='text/csv')


def render_sandbox(root):
    st.info('Изменение входных признаков показывает чувствительность прогноза наблюдательной модели. '
            'Оно не оценивает причинный эффект лечения, снижения веса или изменения анализов.')
    if not st.checkbox('Открыть интерактивный анализ', key='open_sandbox'):
        st.caption('Фиксированные модели Кокса, проверяемые скрининговые правила и калькуляция добавочной стоимости.')
        return
    service = research_service(root)
    row = {feature: np.nan for feature in PREDICTOR_FEATURES}
    a, b, c = st.columns(3)
    row['age'] = a.number_input('Возраст, лет', 18., 100., 40., key='sandbox_age')
    row['gender'] = 1. if b.selectbox('Пол', ['Мужской', 'Женский']) == 'Мужской' else 2.
    row['bmi'] = c.number_input('BMI, кг/м²', 12., 60., 25., key='sandbox_bmi')
    row['sbp'] = a.number_input('Систолическое АД', 70., 250., 120.)
    row['dbp'] = b.number_input('Диастолическое АД', 40., 150., 80.)
    family = c.selectbox('Семейный анамнез диабета', ['Неизвестен', 'Нет', 'Да, родство не уточнено', 'Да, первая степень'])
    row['family_history_diabetes'] = np.nan if family == 'Неизвестен' else float(family != 'Нет')
    asian = st.checkbox('Подтверждено азиатское происхождение', value=True)
    therapy = st.selectbox('Антигипертензивная терапия', ['Неизвестно', 'Нет', 'Да'])
    preset = st.selectbox('Схема стоимости', ['Относительные единицы', 'Справочные тарифы KZT'])
    tariffs = read_json(root / 'configs/reference_tariffs.json')
    kzt = preset == 'Справочные тарифы KZT'
    available = 47 if kzt else 63
    acquired = st.multiselect('Уже известные лабораторные показатели',
                              [f for i, f in enumerate(CORE_ACQUIRABLE_FEATURES) if available & (1 << i)])
    defaults = [5., 5., 1.5, 25., 70., 4.5]
    for i, feature in enumerate(CORE_ACQUIRABLE_FEATURES):
        if feature in acquired:
            row[feature] = st.number_input(feature, min_value=0., value=defaults[i], key='value_' + feature)
    mask = sum(1 << i for i, f in enumerate(CORE_ACQUIRABLE_FEATURES) if f in acquired)
    mode = st.radio('Образец крови', ['Новый забор', 'Уже взятый образец'], horizontal=True)
    unit = 'KZT' if kzt else 'ед.'
    if kzt:
        draw = float(tariffs['draw']) if mode == 'Новый забор' else 0.
        test_prices = tuple(float(tariffs['tests'][f]['price'] or 1750) for f in CORE_ACQUIRABLE_FEATURES)
        st.caption(f"INVITRO, Алматы; проверено {tariffs['checked_on']}. Забор: {draw:g} KZT. "
                   'CCR исключен: соответствие клиническому тесту не установлено. Это справочные розничные тарифы, '
                   'не подтверждение экономической эффективности; на страницах поставщика встречаются разные даты и сборы за забор.')
        with st.expander('Источники тарифов'):
            for feature, spec in tariffs['tests'].items():
                if spec['source']:
                    st.markdown(f"[{feature}: {spec['price']} KZT]({spec['source']})")
            st.markdown(f"[Забор: {tariffs['draw']} KZT]({tariffs['draw_source']})")
    else:
        draw = float(st.selectbox('Стоимость первого забора', [0, 1, 5, 20])) if mode == 'Новый забор' else 0.
        test_prices = (1.,) * 6
    costs = MarginalCosts(test_prices, draw)
    criterion = CRITERIA[st.selectbox('Критерий политики', list(CRITERIA), index=3)]
    k = st.selectbox('Валидационный бюджет для выбора веса', [1, 2, 3, 4, 6], index=2)
    weight = service.multi_selection['budgets'][str(k)]['adaptive']['weight']
    budget = st.number_input(f'Общий бюджет, {unit}', min_value=0., value=float(draw + min(k, available.bit_count()) * test_prices[0]))
    penalty = None
    if criterion != 'multiobjective':
        reference = service.selection['draw_0']['policies'][criterion + '__myopic']['parameter']
        default = (reference or 0.) / test_prices[0]
        penalty = st.number_input('Штраф за единицу стоимости (исследовательский)', min_value=0., value=default, format='%.8f')
        st.caption('Штраф здесь задается вручную. KZT-сценарий не проходил отдельную валидационную настройку.')
    frame = pd.DataFrame([row], columns=PREDICTOR_FEATURES)
    prepared = service.prepare(frame)
    addition, scores = service.next_action(prepared, mask, criterion, costs, available=available,
                                          k=k, budget=budget, penalty=penalty, weight=weight)
    risk = float(prepared.subset_risks[0, mask])
    left, right = st.columns(2)
    left.metric('3-летний риск модели', f'{risk:.2%}')
    right.metric('Стоимость выбранного набора', f'{float(costs.total(mask)):.0f} {unit}')
    if float(costs.total(mask)) > budget:
        st.warning('Стоимость уже известных анализов превышает заданный общий бюджет.')
    action = 'STOP' if not addition else ' + '.join(f for i, f in enumerate(CORE_ACQUIRABLE_FEATURES) if addition & (1 << i))
    st.write('Следующее действие статистической политики: **' + action + '**')
    st.dataframe(pd.DataFrame(scores), hide_index=True, width='stretch')
    render_probabilities(service, prepared, frame, mask, available)
    rows = []
    for guideline, source in SOURCES.items():
        result = screening_triggers(frame, guideline, asian_ancestry=asian,
                                    first_degree_family=None if family in ('Неизвестен', 'Да, родство не уточнено') else [int(family == 'Да, первая степень')],
                                    antihypertensive=None if therapy == 'Неизвестно' else [int(therapy == 'Да')])
        fired = [name for name, value in result['reasons'].items() if value[0]]
        require = bool(result['require_fpg'][0])
        rows.append({'Правило': guideline, 'Сработало': ', '.join(fired) or 'Нет наблюдаемого триггера',
                     'FPG требуется для скрининга': require, 'Добавочная стоимость FPG': float(costs.delta(mask, 1)) if require else 0.})
        st.markdown(f'[{guideline}: первичный источник]({source})')
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption('Показана доступная часть показаний к скринингу, а не полный клинический протокол. '
               'Отсутствие триггера не означает отсутствие показаний; HbA1c и ОГТТ не моделируются.')
    varying = st.selectbox('Признак для кривой чувствительности', ['bmi', 'age', 'sbp'] + acquired)
    center = float(row[varying])
    grid = np.linspace(max(.1, center * .8), center * 1.2, 21)
    varied = pd.concat([frame] * len(grid), ignore_index=True)
    varied[varying] = grid
    predictions = service.prepare(varied).subset_risks[:, mask]
    st.line_chart(pd.DataFrame({'3-летний риск': predictions}, index=pd.Index(grid, name=varying)))
