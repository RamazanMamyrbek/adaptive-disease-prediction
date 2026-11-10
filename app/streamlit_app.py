from pathlib import Path
import runpy
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2d_aaf.demo.research_ui import CRITERIA, render_replay, render_sandbox

st.set_page_config(page_title="Adaptive acquisition research", layout="wide")
replay, sandbox = st.tabs([
    "Ретроспективный разбор когорт (Cohort Replay v1 & v2)",
    "Анализ чувствительности прогноза модели (Model Sensitivity & Guideline Sandbox)",
])
with replay:
    cohort_name = st.selectbox("Когорта", ["Rich Healthcare", "NAGALA"])
    policy_name = st.selectbox("Версия / критерий", ["v1 — фиксированная политика"] + list(CRITERIA))
    if cohort_name == "Rich Healthcare" and policy_name.startswith("v1"):
        runpy.run_path(str(ROOT / "app/replay_v1.py"), run_name="__main__")
    else:
        try:
            render_replay(ROOT, "internal" if cohort_name == "Rich Healthcare" else "external",
                          "v1" if policy_name.startswith("v1") else CRITERIA[policy_name])
        except FileNotFoundError as exc:
            st.error(f"Сначала выполните обучение и расчеты: {exc.filename}")
with sandbox:
    try:
        render_sandbox(ROOT)
    except FileNotFoundError as exc:
        st.error(f"Недоступен локальный исследовательский артефакт: {exc.filename}")
