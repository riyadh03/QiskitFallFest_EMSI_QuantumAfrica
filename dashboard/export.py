"""Minimal, non-destructive export for the CropLab dashboard.

Fills the gaps in the saved results without changing any scientific computation and without overwriting any
existing output. Everything is written to outputs/dashboard/dashboard_export.json:

1. Submitted fusion on validation: predictions, metrics and timings (no fusion validation predictions were saved).
   Uses challenge.submission.Submission exactly as submitted, plus the two branches' decision margins.
2. alpha scan (101 points), via challenge.fusion.tune_fusion_weights.
3. Paired Gaussian-noise robustness of alpha = 0 (RBF), 0.53 (fusion) and 1 (QK) with mean AND standard deviation.
   Same protocol as challenge.fusion.evaluate_fusion_robustness (sigma x training std, 5 draws, seed 2026
   re-seeded per alpha); that function only returns means.
4. Circuit resources (depth, two-qubit gates) of the circuit designs listed in outputs/circuit_benchmark.json,
   rebuilt from the definitions in challenge/circuits.py with placeholder angles (resources do not depend on data).

    python -m dashboard.export        (~1 min, run from the repository root)
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from sklearn.metrics import accuracy_score, f1_score

from challenge.data import ROOT, load_data
from challenge.fusion import softmax, tune_fusion_weights
from challenge.metrics import score
from challenge.qkernel import LABELS, QUBITS, MatchedRBF, QuantumKernelSVM
from challenge.submission import Submission

OUT = ROOT / 'outputs/dashboard'


# ------------------------------------------------------------------ circuit resources
def _chain(qc, x, a, g, pairs):
    for i in range(QUBITS):
        qc.rz(2 * a * x[i], i)
    for i, j in pairs:
        qc.rzz(2 * g * x[i] * x[j], i, j)


def circuit_for(name: str):
    """Rebuild a benchmarked design from challenge/circuits.py; None when the design is not defined there."""
    x = np.linspace(0.1, 0.8, QUBITS)
    linear = [(i, i + 1) for i in range(QUBITS - 1)]
    n = name.lower()
    qc = QuantumCircuit(QUBITS)
    if 'zzfeaturemap' in n:
        return ZZFeatureMap(QUBITS, reps=1, entanglement='linear').assign_parameters(x).decompose()
    if 'ry-rx' in n:
        for i in range(QUBITS):
            qc.ry(0.2 * x[i], i); qc.rx(0.2 * x[i], i)
        for i, j in linear:
            qc.rzz(0.02 * x[i] * x[j], i, j)
        return qc
    if 're-uploading' in n:
        qc.h(range(QUBITS)); _chain(qc, x, 0.2, 0.01, linear)
        for i in range(QUBITS):
            qc.ry(np.pi / 4, i)
        _chain(qc, x, 0.2, 0.01, linear)
        return qc
    if 'star' in n:
        return None  # in outputs/circuit_benchmark.json but not defined in the current challenge/circuits.py
    qc.h(range(QUBITS))
    if 'circular' in n or 'ring' in n:
        _chain(qc, x, 0.2, 0.01, [(i, (i + 1) % QUBITS) for i in range(QUBITS)])
    elif 'all-to-all' in n:
        _chain(qc, x, 0.2, 0.005, [(i, j) for i in range(QUBITS) for j in range(i + 1, QUBITS)])
    elif 'no entanglement' in n:
        _chain(qc, x, 0.2, 0.0, [])
    else:  # current linear chain and the bandwidth variants (same structure, different a)
        _chain(qc, x, 0.2, 0.01, linear)
    return qc


def circuit_resources():
    rows = []
    for r in json.loads((ROOT / 'outputs/circuit_benchmark.json').read_text()):
        qc = circuit_for(r['Architecture'])
        if qc is None:
            rows.append(dict(architecture=r['Architecture'], depth=None, two_qubit_gates=None, total_gates=None,
                             note='Design not defined in the current challenge/circuits.py; resources not recorded.'))
            continue
        rows.append(dict(architecture=r['Architecture'], depth=int(qc.depth()),
                         two_qubit_gates=int(sum(len(i.qubits) == 2 for i in qc.data)),
                         total_gates=int(len(qc.data)), note=None))
    return rows


def draw_feature_map():
    """Picture of the submitted feature map (symbolic x_i), drawn by Qiskit from challenge.qkernel.feature_map."""
    import matplotlib
    matplotlib.use('Agg')
    from challenge.qkernel import feature_map
    cfg = json.loads((ROOT / 'configs/final_quantum_kernel.json').read_text())
    qc, _ = feature_map(cfg['a'], cfg['gamma'], cfg['reps'])
    fig = qc.draw('mpl', style={'backgroundcolor': '#FFFFFF', 'fontsize': 11}, fold=-1)
    fig.savefig(OUT / 'qk_feature_map.png', dpi=160, bbox_inches='tight', facecolor='white')


# ------------------------------------------------------------------ fusion on validation
def main():
    t0 = perf_counter()
    tables, _, manifest, _ = load_data()
    F = manifest['features']
    train, val = tables['train'], tables['validation']
    X_tr, y_tr = train[F].to_numpy(), train.target.to_numpy()
    X_va, y_va = val[F].to_numpy(), val.target.to_numpy()
    labels = list(LABELS)

    # Submitted model, timed exactly as the organizers would call it.
    sub = Submission()
    s = perf_counter(); sub.fit(X_tr, y_tr); fit_s = perf_counter() - s
    s = perf_counter(); pred_fusion = sub.predict(X_va); pred_s = perf_counter() - s

    # Branch margins (same configs as the submission) for the alpha scan and the branch predictions.
    qk = QuantumKernelSVM(sub.a, sub.gamma_qk, sub.reps, sub.C_qk).fit(X_tr, y_tr)
    rbf = MatchedRBF(sub.gamma_rbf, sub.C_rbf).fit(X_tr, y_tr)
    d_qk = qk.svm.decision_function(qk.kernel_rows(X_va))
    classes = qk.svm.classes_
    pred_qk_argmax = classes[d_qk.argmax(axis=1)]
    pred_qk_vote = qk.svm.predict(qk.kernel_rows(X_va))
    alpha_scan = tune_fusion_weights(qk, rbf, X_va, y_va).to_dict(orient='records')

    # Paired robustness with mean and sd.
    std = X_tr.std(axis=0)
    sigmas, draws, seed = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5], 5, 2026
    robustness = []
    for name, alpha in [('matched_rbf', 0.0), ('hybrid_fusion', 0.53), ('quantum_kernel', 1.0)]:
        rng = np.random.default_rng(seed)
        for sig in sigmas:
            f1s, accs = [], []
            for _ in range(draws if sig > 0 else 1):
                V = X_va + sig * std * rng.standard_normal(X_va.shape)
                dq = qk.svm.decision_function(qk.kernel_rows(V))
                dr = rbf.svm.decision_function(rbf.prep.transform(V))
                p = classes[(alpha * softmax(dq) + (1 - alpha) * softmax(dr)).argmax(axis=1)]
                f1s.append(f1_score(y_va, p, average='macro')); accs.append(accuracy_score(y_va, p))
            robustness.append(dict(model=name, alpha=alpha, sigma=sig, draws=len(f1s),
                                   macro_f1_mean=float(np.mean(f1s)), macro_f1_sd=float(np.std(f1s)),
                                   accuracy_mean=float(np.mean(accs)), accuracy_sd=float(np.std(accs))))

    out = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        environment=dict(python=platform.python_version(), machine=platform.machine(), processor=platform.processor(),
                         qiskit=version('qiskit'), scikit_learn=version('scikit-learn')),
        split='validation (150 fields, 50 per class)',
        fusion=dict(alpha=sub.alpha,
                    config=dict(qk=dict(a=sub.a, gamma=sub.gamma_qk, reps=sub.reps, C=sub.C_qk),
                                rbf=dict(gamma=sub.gamma_rbf, C=sub.C_rbf)),
                    metrics=score(y_va, pred_fusion, labels), fit_seconds=fit_s, predict_150_seconds=pred_s,
                    deterministic=True, seeds_note='Deterministic: seeds 0, 1 and 2 give the identical result.'),
        quantum_kernel_decision_rules=dict(
            vote_predict=score(y_va, pred_qk_vote, labels), argmax_margins=score(y_va, pred_qk_argmax, labels),
            differing_fields=[int(i) for i in np.flatnonzero(pred_qk_vote != pred_qk_argmax)],
            note='svm.predict uses one-vs-one votes; a 3-way vote tie is given to the first class. '
                 'The fusion at alpha = 1 equals the argmax rule.'),
        validation_predictions=dict(sample_id=val.sample_id.tolist(), target=y_va.tolist(),
                                    hybrid_fusion=pred_fusion.tolist(), quantum_kernel_argmax=pred_qk_argmax.tolist(),
                                    quantum_kernel_vote=pred_qk_vote.tolist(), matched_rbf=rbf.predict(X_va).tolist()),
        alpha_scan=alpha_scan,
        robustness_paired=dict(protocol='Gaussian noise sigma x training std of each raw feature, added to validation; '
                                        'models trained on clean data; 5 draws per sigma; generator re-seeded (2026) '
                                        'for every model so all models see identical noisy copies.',
                               rows=robustness),
        circuit_resources=circuit_resources(),
        seconds=round(perf_counter() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'dashboard_export.json').write_text(json.dumps(out, indent=2, default=float))
    draw_feature_map()
    m = out['fusion']['metrics']
    print(f"fusion validation macro-F1 {m['macro_f1']:.4f}  accuracy {m['accuracy']:.4f}  "
          f"fit {fit_s:.1f}s  predict {pred_s:.2f}s")
    print(f"wrote {OUT / 'dashboard_export.json'} in {out['seconds']} s")


if __name__ == '__main__':
    main()
