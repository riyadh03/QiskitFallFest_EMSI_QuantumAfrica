"""Cached loaders for the saved experiment results. The dashboard never trains a model or runs a circuit.

Every number shown in the app comes from one of the files listed in SOURCES. Missing files return None and the
pages show "Not recorded" instead of failing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    'qk_summary': 'outputs/qk/summary.json',                    # QK, γ=0 ablation, matched RBF, reference (3 seeds)
    'qk_val_pred': 'outputs/qk/validation_predictions.csv',
    'reliability': 'outputs/qk/reliability.json',               # Gaussian noise (raw scale) + band drop
    'grid_quantum': 'outputs/qk/grid_quantum.csv',
    'grid_rbf': 'outputs/qk/grid_rbf.csv',
    'starter': 'results/summary.json',                          # organizer-supplied starter QNN / MLP / reference
    'alternatives': 'outputs/benchmark/alternatives_summary.json',
    'circuits': 'outputs/circuit_benchmark.json',
    'export': 'outputs/dashboard/dashboard_export.json',        # fusion predictions, α scan, paired robustness
    'audit': 'outputs/audit/validation_audit.json',
    'hw_summary': 'outputs/qk/hardware/hardware_summary.json',
    'hw_fusion': 'outputs/qk/hardware/fusion_hardware.json',
    'shots': 'outputs/qk/hardware/shots_study.json',
    'class_counts': 'outputs/class_counts.csv',
    'manifest': 'data/manifest.json',
    'final_config': 'configs/final_quantum_kernel.json',
    'reference_config': 'configs/reference.json',
}


@st.cache_data(show_spinner=False)
def load(key):
    p = ROOT / SOURCES[key]
    if not p.exists():
        return None
    if p.suffix == '.json':
        return json.loads(p.read_text(encoding='utf-8'))
    if p.suffix == '.csv':
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def hardware_reports():
    """All analysed IBM jobs, oldest first, with a flag for the ones pooled in the summaries (twirled)."""
    out = []
    for f in sorted((ROOT / 'outputs/qk/hardware').glob('report_*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        out.append(r)
    order = {'dc01gtinj2vs73dda5l0': 0, 'dc01ihanj2vs73dda82g': 1, 'dc027sls2omc7388l0q0': 2}
    return sorted(out, key=lambda r: order.get(r['job_id'], 99))


def path(key):
    return SOURCES[key]


def exists(key):
    return (ROOT / SOURCES[key]).exists()


def asset(rel):
    p = ROOT / rel
    return str(p) if p.exists() else None


# ---------------------------------------------------------------------------------------------- helpers
def qk_run(model, seed=0):
    s = load('qk_summary')
    if not s:
        return None
    for r in s['runs']:
        if r['model'] == model and r['seed'] == seed:
            return r
    return None


def qk_seed_stats(model):
    """Mean/sd of macro-F1 over seeds 0-2 and mean timings from outputs/qk/summary.json."""
    s = load('qk_summary')
    rs = [r for r in s['runs'] if r['model'] == model] if s else []
    if not rs:
        return None
    f1 = [r['metrics']['macro_f1'] for r in rs]
    return dict(seeds=sorted(r['seed'] for r in rs), f1_mean=float(np.mean(f1)), f1_sd=float(np.std(f1)),
                fit=float(np.mean([r['fit_seconds'] for r in rs])),
                predict=float(np.mean([r['predict_150_seconds'] for r in rs])))


def starter_model(model):
    s = load('starter')
    if not s:
        return None
    for m in s['metrics']:
        if m['model'] == model and m['train'] == 914:
            return m
    return None


def alternative(prefix):
    a = load('alternatives')
    if not a:
        return None
    for r in a['architectures']:
        if r['name'].startswith(prefix):
            return r
    return None


def recall_from_cm(cm):
    cm = np.asarray(cm, float)
    return (np.diag(cm) / cm.sum(axis=1)).tolist()


def precision_from_cm(cm):
    cm = np.asarray(cm, float)
    col = cm.sum(axis=0)
    return np.divide(np.diag(cm), col, out=np.zeros(3), where=col > 0).tolist()
