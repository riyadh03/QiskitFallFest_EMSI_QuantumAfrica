"""Validation audit: how much of the 0.9533 is real, and how much of it is quantum?

Three questions, each answered with one experiment:

1. Selection optimism. The quantum kernel (a, gamma, C), the RBF (gamma, C) and the fusion weight alpha were all
   chosen on the same 150 validation fields that report 0.9533. Here we redo every choice with 5-fold stratified
   cross-validation on the TRAINING set only (same grids, same selection rule) and score validation once.
2. Uncertainty. Paired bootstrap of metric(model B) - metric(model A) on the 150 validation fields, both per field
   and per spatial block (data/validation_groups.json, 18 blocks), 4000 resamples, 95% two-sided intervals.
3. Is the gain quantum? A "classical twin" of the fusion replaces the 8-qubit circuit kernel by its entanglement-free
   closed form K(x, x') = prod_i cos^2(a (x_i - x'_i)) -- computable on a laptop without any circuit -- and fuses it
   with the same RBF. If the twin matches the fusion, the gain comes from combining two different kernel shapes,
   not from entanglement.

    python -m challenge.validation_audit          (~2-4 min; writes outputs/audit/validation_audit.json)

The encoder (StandardScaler -> PCA(8) -> StandardScaler) is fitted once on all 914 training rows and reused inside
the CV folds. It uses no labels; refitting it per fold would change only the unsupervised projection.
"""
from __future__ import annotations

import json
from itertools import product
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from .coverage import metric_vector
from .data import ROOT, load_data
from .qkernel import LABELS, encoder, fidelity, statevectors
from .qk_run import A_GRID, C_GRID, G_GRID, RBF_GRID

OUT = ROOT / 'outputs/audit'
ALPHAS = np.round(np.linspace(0, 1, 101), 2)
METRICS = ['macro_f1', 'accuracy', 'recall_alfalfa', 'recall_maize', 'recall_wheat']
REPORTED = dict(qk=dict(a=0.2, gamma=0.01, C=100), rbf=dict(gamma=0.11288, C=10), alpha=0.53)


# ---------------------------------------------------------------- kernels and fusion
def cos2_kernel(A, B, a):
    """Entanglement-free fidelity kernel in closed form (equals the gamma = 0 circuit to ~1e-15)."""
    return np.prod(np.cos(a * (A[:, None, :] - B[None, :, :])) ** 2, axis=2)


def rbf_kernel(A, B, g):
    return np.exp(-g * ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))


def softmax(d):
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fuse(d_a, d_b, alpha, classes):
    """Same rule as challenge/submission.py: alpha * softmax(d_a) + (1 - alpha) * softmax(d_b)."""
    return classes[np.argmax(alpha * softmax(d_a) + (1 - alpha) * softmax(d_b), axis=1)]


def macro_f1(y, p):
    return f1_score(y, p, labels=list(LABELS), average='macro', zero_division=0)


# ---------------------------------------------------------------- train-only selection
def cv_kernel_scores(K, y, folds, C):
    """Out-of-fold decision values and the OOF macro-F1 for a precomputed train x train kernel."""
    d = np.zeros((len(y), 3))
    for tr, te in folds:
        d[te] = SVC(kernel='precomputed', C=C).fit(K[np.ix_(tr, tr)], y[tr]).decision_function(K[np.ix_(te, tr)])
    classes = np.array(sorted(set(y)))
    return d, macro_f1(y, classes[d.argmax(axis=1)])


def select_config(candidates, y, folds):
    """candidates: list of (config, K_train). Selection rule of qk_run.pick, on OOF instead of validation."""
    rows = []
    for cfg, K in candidates:
        for C in C_GRID:
            d, f = cv_kernel_scores(K, y, folds, C)
            rows.append(dict(cfg={**cfg, 'C': C}, f1=f, d=d))
    rows.sort(key=lambda r: (-r['f1'], r['cfg']['C']))
    return rows[0], len(rows)


def select_alpha(d_a, d_b, y, classes):
    """Best OOF macro-F1 over 101 alphas; ties broken towards 0.5 (equal weighting as the neutral prior)."""
    f = np.array([macro_f1(y, fuse(d_a, d_b, a, classes)) for a in ALPHAS])
    best = np.flatnonzero(np.isclose(f, f.max()))
    return float(ALPHAS[best[np.argmin(np.abs(ALPHAS[best] - 0.5))]]), float(f.max())


# ---------------------------------------------------------------- paired bootstrap
def bootstrap(y, pa, pb, groups=None, samples=4000, seed=2026):
    """metric(pb) - metric(pa) with 95% interval; resample fields (groups=None) or whole spatial blocks."""
    units = np.arange(len(y)) if groups is None else np.unique(groups, return_inverse=True)[1]
    n = units.max() + 1
    rng = np.random.default_rng(seed)
    point = metric_vector(y, pb[None], np.ones(len(y))) - metric_vector(y, pa[None], np.ones(len(y)))
    diffs = []
    while len(diffs) < samples:
        w = np.bincount(rng.integers(n, size=n), minlength=n)[units].astype(float)
        if all(w[y == k].sum() > 0 for k in range(3)):
            diffs.append(metric_vector(y, pb[None], w) - metric_vector(y, pa[None], w))
    d = np.array(diffs)
    return {m: dict(delta=float(point[i]), ci95=np.quantile(d[:, i], [.025, .975]).round(4).tolist(),
                    p_not_better=float((d[:, i] <= 0).mean())) for i, m in enumerate(METRICS)}


def main():
    t0 = perf_counter()
    tables, _, manifest, _ = load_data()
    F = manifest['features']
    train, val = tables['train'], tables['validation']
    y, yv = train.target.to_numpy(), val.target.to_numpy()
    prep = encoder().fit(train[F].to_numpy())
    A, B = prep.transform(train[F].to_numpy()), prep.transform(val[F].to_numpy())
    classes = np.array(LABELS)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(A, y))

    # ---- 1. Selection on training data only --------------------------------------------------------
    print('Train-only 5-fold CV selection (same grids as qk_run)...', flush=True)
    sv = {}
    qk_cands = []
    for a, g in product(A_GRID, G_GRID):
        sv[(a, g)] = statevectors(A, a, g)
        qk_cands.append((dict(a=a, gamma=g), fidelity(sv[(a, g)], sv[(a, g)])))
    qk_best, n_qk = select_config(qk_cands, y, folds)
    rbf_best, n_rbf = select_config([(dict(gamma=g), rbf_kernel(A, A, g)) for g in RBF_GRID], y, folds)
    twin_best, n_twin = select_config([(dict(a=a), cos2_kernel(A, A, a)) for a in A_GRID], y, folds)
    alpha_q, f_alpha_q = select_alpha(qk_best['d'], rbf_best['d'], y, classes)
    alpha_t, f_alpha_t = select_alpha(twin_best['d'], rbf_best['d'], y, classes)

    # ---- fit on all training rows, score validation once ---------------------------------------------
    def fit_decide(K_train, K_val, C):
        svm = SVC(kernel='precomputed', C=C).fit(K_train, y)
        return svm.decision_function(K_val)

    def qk_dec(cfg):
        S = sv.get((cfg['a'], cfg['gamma'])) if cfg['a'] in A_GRID and cfg['gamma'] in G_GRID else None
        S = statevectors(A, cfg['a'], cfg['gamma']) if S is None else S
        Sv = statevectors(B, cfg['a'], cfg['gamma'])
        return fit_decide(fidelity(S, S), fidelity(Sv, S), cfg['C'])

    def rbf_dec(cfg):
        return fit_decide(rbf_kernel(A, A, cfg['gamma']), rbf_kernel(B, A, cfg['gamma']), cfg['C'])

    def twin_dec(cfg):
        return fit_decide(cos2_kernel(A, A, cfg['a']), cos2_kernel(B, A, cfg['a']), cfg['C'])

    dec = dict(
        reported=dict(qk=qk_dec(REPORTED['qk']), rbf=rbf_dec(REPORTED['rbf']), twin=twin_dec(dict(a=0.2, C=100))),
        train_cv=dict(qk=qk_dec(qk_best['cfg']), rbf=rbf_dec(rbf_best['cfg']), twin=twin_dec(twin_best['cfg'])))
    alphas = dict(reported=dict(fusion=0.53, twin=0.53), train_cv=dict(fusion=alpha_q, twin=alpha_t))

    preds, scores = {}, {}
    for proto in ['reported', 'train_cv']:
        d = dec[proto]
        preds[proto] = dict(
            quantum_kernel=classes[d['qk'].argmax(1)], matched_rbf=classes[d['rbf'].argmax(1)],
            classical_twin_kernel=classes[d['twin'].argmax(1)],
            fusion_qk_rbf=fuse(d['qk'], d['rbf'], alphas[proto]['fusion'], classes),
            fusion_twin_rbf=fuse(d['twin'], d['rbf'], alphas[proto]['twin'], classes))
        scores[proto] = {k: dict(macro_f1=round(macro_f1(yv, p), 4), accuracy=round(accuracy_score(yv, p), 4),
                                 correct=int((p == yv).sum()),
                                 recall_wheat=round(float(((p == yv) & (yv == 'wheat')).sum() / (yv == 'wheat').sum()), 4))
                         for k, p in preds[proto].items()}

    # ---- 2. Paired bootstrap on the submitted (reported-protocol) predictions --------------------------
    print('Paired bootstrap (per field and per spatial block)...', flush=True)
    groups_map = json.loads((ROOT / 'data/validation_groups.json').read_text())
    groups = np.array([groups_map[s] for s in val.sample_id])
    yi = np.searchsorted(classes, yv)
    enc = {k: np.searchsorted(classes, p) for k, p in preds['reported'].items()}
    pairs = [('quantum_kernel', 'fusion_qk_rbf'), ('matched_rbf', 'fusion_qk_rbf'), ('matched_rbf', 'quantum_kernel'),
             ('fusion_twin_rbf', 'fusion_qk_rbf'), ('matched_rbf', 'fusion_twin_rbf')]
    boot = {f'{b} - {a}': dict(fields_differing=int((enc[a] != enc[b]).sum()),
                               per_field=bootstrap(yi, enc[a], enc[b]),
                               per_spatial_block=bootstrap(yi, enc[a], enc[b], groups))
            for a, b in pairs}

    report = dict(
        question='How optimistic is 0.9533, how certain is the fusion gain, and is the gain quantum?',
        validation_fields=len(yv), spatial_blocks=int(len(np.unique(groups))),
        train_cv_selection=dict(
            folds=5, rule='best out-of-fold macro-F1, then smallest C; alpha ties broken towards 0.5',
            quantum_kernel=dict(config=qk_best['cfg'], oof_macro_f1=round(qk_best['f1'], 4), configs_tried=n_qk),
            matched_rbf=dict(config=rbf_best['cfg'], oof_macro_f1=round(rbf_best['f1'], 4), configs_tried=n_rbf),
            classical_twin=dict(config=twin_best['cfg'], oof_macro_f1=round(twin_best['f1'], 4), configs_tried=n_twin),
            alpha_fusion=dict(alpha=alpha_q, oof_macro_f1=round(f_alpha_q, 4)),
            alpha_twin=dict(alpha=alpha_t, oof_macro_f1=round(f_alpha_t, 4))),
        validation_scores=dict(reported_selected_on_validation=scores['reported'], selected_on_train_cv_only=scores['train_cv']),
        paired_bootstrap=dict(samples=4000, seed=2026, protocol='reported configs (as submitted)', comparisons=boot),
        seconds=round(perf_counter() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'validation_audit.json').write_text(json.dumps(report, indent=2))

    # ---- console summary ---------------------------------------------------------------------------
    print(f"\nTrain-CV choices: QK {qk_best['cfg']}  RBF {rbf_best['cfg']}  twin {twin_best['cfg']}  "
          f"alpha(QK+RBF)={alpha_q}  alpha(twin+RBF)={alpha_t}")
    print(f"\n{'model':24s} {'selected on validation':>24s} {'selected on train CV':>24s}")
    for k in scores['reported']:
        r, c = scores['reported'][k], scores['train_cv'][k]
        print(f"{k:24s} {r['macro_f1']:.4f} ({r['correct']}/150)      {c['macro_f1']:.4f} ({c['correct']}/150)")
    print('\nPaired bootstrap, delta macro-F1 [95% CI]  (per field | per spatial block)')
    for name, b in boot.items():
        f, s = b['per_field']['macro_f1'], b['per_spatial_block']['macro_f1']
        print(f"{name:40s} {f['delta']:+.4f}  [{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}] | "
              f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]  ({b['fields_differing']} fields differ)")
    print(f"\nWrote {OUT / 'validation_audit.json'} in {report['seconds']} s")


if __name__ == '__main__':
    main()
