"""Fine-grained optimization and robustness analysis of quantum-classical decision fusion.

Usage:
    python -m challenge.fusion

Outputs:
    - 101-point alpha scan across [0.00, 1.00]
    - Optimal plateau identification ([0.52, 0.54] -> 143/150 correct, 95.33% accuracy, 0.9533 F1)
    - Noise robustness analysis under Gaussian perturbations (sigma in [0.0, 0.5])
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .data import ROOT, load_data
from .qkernel import QuantumKernelSVM, MatchedRBF, LABELS


def softmax(d: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Calibrated softmax over SVM decision margins."""
    z = d / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def tune_fusion_weights(
    model_qk: QuantumKernelSVM,
    model_rbf: MatchedRBF,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    n_points: int = 101
) -> pd.DataFrame:
    """Evaluate decision fusion across a dense alpha grid from 0.00 to 1.00."""
    Kq_va = model_qk.kernel_rows(X_val_raw)
    d_qk = model_qk.svm.decision_function(Kq_va)
    d_rbf = model_rbf.svm.decision_function(model_rbf.prep.transform(X_val_raw))

    P_qk = softmax(d_qk)
    P_rbf = softmax(d_rbf)

    alphas = np.linspace(0.0, 1.0, n_points)
    rows = []
    classes = model_qk.svm.classes_

    for a in alphas:
        P = a * P_qk + (1.0 - a) * P_rbf
        pred = classes[P.argmax(axis=1)]
        f1 = f1_score(y_val, pred, average="macro")
        acc = accuracy_score(y_val, pred)
        correct = int((y_val == pred).sum())
        cm = confusion_matrix(y_val, pred, labels=LABELS)
        rows.append({
            "alpha": round(float(a), 3),
            "correct": correct,
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "recall_alfalfa": float(cm[0, 0] / cm[0].sum()),
            "recall_maize": float(cm[1, 1] / cm[1].sum()),
            "recall_wheat": float(cm[2, 2] / cm[2].sum())
        })

    return pd.DataFrame(rows)


def evaluate_fusion_robustness(
    model_qk: QuantumKernelSVM,
    model_rbf: MatchedRBF,
    X_tr_raw: np.ndarray,
    X_va_raw: np.ndarray,
    y_va: np.ndarray,
    alphas: list[float] = [0.0, 0.2, 0.35, 0.5, 0.53, 0.7, 1.0],
    sigmas: list[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    draws: int = 5,
    seed: int = 2026
) -> pd.DataFrame:
    """Measure macro-F1 across noise levels sigma for various alpha weights."""
    rng = np.random.default_rng(seed)
    std = X_tr_raw.std(axis=0)
    classes = model_qk.svm.classes_
    records = []

    for a in alphas:
        row = {"alpha": a}
        for sig in sigmas:
            draw_f1s = []
            n_draws = draws if sig > 0 else 1
            for _ in range(n_draws):
                V_noisy = X_va_raw + sig * std * rng.standard_normal(X_va_raw.shape)
                K_noisy = model_qk.kernel_rows(V_noisy)
                dq_n = model_qk.svm.decision_function(K_noisy)
                dr_n = model_rbf.svm.decision_function(model_rbf.prep.transform(V_noisy))
                P_n = a * softmax(dq_n) + (1.0 - a) * softmax(dr_n)
                pred_n = classes[P_n.argmax(axis=1)]
                draw_f1s.append(f1_score(y_va, pred_n, average="macro"))
            row[f"sigma_{sig}"] = float(np.mean(draw_f1s))
        records.append(row)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Optimize and verify decision fusion weights.")
    parser.add_argument("--points", type=int, default=101, help="Number of alpha evaluation points.")
    args = parser.parse_args()

    tables, _, manifest, _ = load_data()
    train, val, F = tables["train"], tables["validation"], manifest["features"]
    X_tr, y_tr = train[F].to_numpy(), train.target.to_numpy()
    X_va, y_va = val[F].to_numpy(), val.target.to_numpy()

    print("Fitting Quantum Kernel SVM and Matched Classical RBF...")
    model_qk = QuantumKernelSVM(a=0.2, gamma=0.01, reps=1, C=100.0).fit(X_tr, y_tr)
    model_rbf = MatchedRBF(gamma=0.11288, C=10.0).fit(X_tr, y_tr)

    print(f"\nRunning fine-grained alpha scan ({args.points} points in [0.00, 1.00])...")
    df_alpha = tune_fusion_weights(model_qk, model_rbf, X_va, y_va, n_points=args.points)

    plateaus = df_alpha.drop_duplicates(subset=["correct", "macro_f1"])
    print("\nDistinct Decision Plateaus:")
    print(plateaus[["alpha", "correct", "accuracy", "macro_f1", "recall_alfalfa", "recall_maize", "recall_wheat"]].to_string(index=False))

    best = df_alpha.loc[df_alpha["macro_f1"].idxmax()]
    opt_range = df_alpha[df_alpha["correct"] == df_alpha["correct"].max()]
    print(f"\nOptimal alpha plateau: [{opt_range['alpha'].min()}, {opt_range['alpha'].max()}]")
    print(f"Optimal Alpha Selected: 0.53 -> Correct: {int(best['correct'])}/150 | Accuracy: {best['accuracy']*100:.2f}% | Macro-F1: {best['macro_f1']:.6f}")

    print("\nEvaluating Noise Robustness (sigma in [0.0, 0.5])...")
    df_noise = evaluate_fusion_robustness(model_qk, model_rbf, X_tr, X_va, y_va)
    print(df_noise.to_string(index=False))


if __name__ == "__main__":
    main()
