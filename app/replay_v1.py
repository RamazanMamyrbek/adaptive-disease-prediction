from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from t2d_aaf.demo.service import (
    FEATURE_LABELS,
    DemoService,
    DisplayValue,
    deterministic_case_number,
)


st.set_page_config(
    page_title="T2D adaptive acquisition",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17211f;
        --muted: #5f6d69;
        --line: #d8e0dd;
        --paper: #ffffff;
        --wash: #f4f7f6;
        --teal: #147d73;
        --coral: #b94f3b;
        --gold: #a87108;
    }
    .stApp { background: var(--wash); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244, 247, 246, 0.94); }
    [data-testid="stSidebar"] { background: #edf2f0; border-right: 1px solid var(--line); }
    .block-container { max-width: 1180px; padding-top: 2.1rem; padding-bottom: 3rem; }
    h1, h2, h3, p, label { letter-spacing: 0 !important; }
    h1 { font-size: 2rem !important; font-weight: 680 !important; }
    h2 { font-size: 1.2rem !important; margin-top: 1.4rem !important; }
    h3 { font-size: 1rem !important; }
    .safety-band {
        border-left: 4px solid var(--coral);
        background: #fff8f6;
        padding: 0.72rem 0.9rem;
        margin: 0.5rem 0 1.2rem 0;
        color: #6b2d22;
        border-radius: 0 6px 6px 0;
    }
    .action-band {
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        background: var(--paper);
        padding: 1rem 0.15rem;
        margin: 1rem 0 1.2rem 0;
    }
    .action-kicker { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; }
    .action-name { color: var(--teal); font-size: 1.38rem; font-weight: 680; margin-top: 0.12rem; }
    .stop-name { color: var(--gold); }
    div[data-testid="stMetric"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.75rem 0.85rem;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    .stButton > button { border-radius: 6px; min-height: 2.6rem; font-weight: 620; }
    .stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }
    .small-note { color: var(--muted); font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading locked research artifacts...")
def load_service() -> DemoService:
    return DemoService.from_project(PROJECT_ROOT)


def value_text(item: DisplayValue) -> str:
    if not item.source_observed or item.value is None:
        return "Source missing"
    if item.feature == "gender":
        return "Male" if item.value == 1.0 else "Female"
    if item.feature == "family_history_diabetes":
        return "Yes" if item.value == 1.0 else "No"
    if item.feature == "age":
        return f"{item.value:.0f}"
    return f"{item.value:.2f}"


def value_table(values: tuple[DisplayValue, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Feature": item.label,
                "Value": value_text(item),
                "Unit": item.unit,
                "Source": "Observed" if item.source_observed else "Missing",
            }
            for item in values
        ]
    )


try:
    service = load_service()
except Exception as exc:
    st.error(f"The local locked demo artifacts could not be loaded: {exc}")
    st.stop()

if "case_number" not in st.session_state:
    st.session_state.case_number = deterministic_case_number(
        20260827,
        service.participant_count,
    )
if "selected_case_number" not in st.session_state:
    st.session_state.selected_case_number = st.session_state.case_number
if "acquired_features" not in st.session_state:
    st.session_state.acquired_features = []
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 20260827


def load_random_case() -> None:
    st.session_state.random_seed += 1
    selected = deterministic_case_number(
        st.session_state.random_seed,
        service.participant_count,
    )
    st.session_state.case_number = selected
    st.session_state.selected_case_number = selected
    st.session_state.acquired_features = []


with st.sidebar:
    st.subheader("Retrospective case")
    st.number_input(
        "Test case number",
        min_value=1,
        max_value=service.participant_count,
        step=1,
        key="selected_case_number",
    )
    if st.button("Load case", use_container_width=True):
        st.session_state.case_number = int(st.session_state.selected_case_number)
        st.session_state.acquired_features = []
        st.rerun()
    st.button("New random case", use_container_width=True, on_click=load_random_case)
    st.divider()
    st.caption("Locked policy")
    st.text("Adaptive acquisition v1")
    st.caption("Prediction horizon")
    st.text("3 years")
    st.caption("Cost scheme")
    st.text("Uniform relative units")

snapshot = service.snapshot(
    int(st.session_state.case_number),
    tuple(st.session_state.acquired_features),
)

st.title("Adaptive T2D acquisition")
st.caption(snapshot.case_label)
st.markdown(
    '<div class="safety-band"><strong>Research simulation</strong> &nbsp; '
    'Retrospective test participant &nbsp; Not clinical advice</div>',
    unsafe_allow_html=True,
)

risk_col, cost_col, count_col = st.columns(3)
risk_col.metric(
    "Model-estimated 3-year risk",
    f"{100.0 * snapshot.current_risk_at_3_years:.2f}%",
)
cost_col.metric("Acquisition cost", f"{snapshot.total_cost:.1f} units")
count_col.metric("Laboratory features acquired", len(snapshot.acquired_features))

if snapshot.stopped:
    st.markdown(
        '<div class="action-band"><div class="action-kicker">Model-selected next action</div>'
        '<div class="action-name stop-name">STOP</div>'
        '<div class="small-note">The locked policy selects no further laboratory feature.</div></div>',
        unsafe_allow_html=True,
    )
else:
    action_label = FEATURE_LABELS[snapshot.next_action]
    st.markdown(
        '<div class="action-band"><div class="action-kicker">Model-selected next feature</div>'
        f'<div class="action-name">{action_label}</div>'
        f'<div class="small-note">Incremental cost: {snapshot.next_action_cost:.1f} relative unit</div></div>',
        unsafe_allow_html=True,
    )
    if st.button(f"Acquire {action_label}", type="primary", use_container_width=True):
        st.session_state.acquired_features = list(
            service.acquire_recommended(snapshot)
        )
        st.rerun()

left, right = st.columns((1.05, 0.95), gap="large")
with left:
    st.subheader("Base features")
    st.dataframe(
        value_table(snapshot.base_values),
        hide_index=True,
        use_container_width=True,
    )

with right:
    st.subheader("Acquired laboratory results")
    if snapshot.acquired_values:
        st.dataframe(
            value_table(snapshot.acquired_values),
            hide_index=True,
            use_container_width=True,
        )
        if any(not item.source_observed for item in snapshot.acquired_values):
            st.warning(
                "A selected historical result is source-missing. Its acquisition is "
                "charged and the locked train-fitted imputation is used by the predictor."
            )
    else:
        st.info("No laboratory result has been acquired for this case.")

st.subheader("Risk trajectory")
trajectory = pd.DataFrame(
    {
        "State": [point.state_label for point in snapshot.risk_trajectory],
        "Model-estimated 3-year risk (%)": [
            100.0 * point.risk_at_3_years for point in snapshot.risk_trajectory
        ],
    }
).set_index("State")
st.line_chart(trajectory, use_container_width=True)
st.caption(
    "Values are model outputs from a retrospective research simulation. "
    "Outcome, follow-up time, site, and participant identifiers are not exposed to the policy or interface."
)
