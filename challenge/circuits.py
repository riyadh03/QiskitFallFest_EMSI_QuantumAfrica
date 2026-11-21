"""Benchmark of quantum circuit designs for the fidelity quantum kernel.

Explores:
    - Feature maps: Custom linear chain vs Qiskit ZZFeatureMap vs Custom RY-RX+RZZ
    - Data re-uploading: Multi-layer encoding with fixed rotation mixing
    - Entanglement topology: Linear vs Circular (Ring) vs Full (All-to-all) vs Star vs None
    - Kernel bandwidth: Scaling factor a in [0.05, 0.40]

Usage:
    python -m challenge.circuits
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

from .data import ROOT, load_data
from .qkernel import encoder, fidelity, MatchedRBF, LABELS, QUBITS


def evaluate_kernel_circuit(
    get_statevector_fn,
    name: str,
    X_tr_pca: np.ndarray,
    X_va_pca: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    C_grid: list[float] = [1.0, 10.0, 100.0, 500.0]
) -> dict:
    """Compute statevectors, fidelity kernel matrix, and evaluate SVM."""
    S_tr = np.array([get_statevector_fn(row) for row in X_tr_pca])
    S_va = np.array([get_statevector_fn(row) for row in X_va_pca])

    K_tr = fidelity(S_tr, S_tr)
    K_va = fidelity(S_va, S_tr)

    best_res = None
    for C in C_grid:
        clf = SVC(kernel="precomputed", C=C).fit(K_tr, y_tr)
        pred = clf.predict(K_va)
        f1 = f1_score(y_va, pred, average="macro")
        acc = accuracy_score(y_va, pred)
        cm = confusion_matrix(y_va, pred, labels=LABELS)
        rec_wheat = cm[2, 2] / cm[2].sum()
        res = {
            "name": name,
            "C": C,
            "correct": int((y_va == pred).sum()),
            "acc": float(acc),
            "f1": float(f1),
            "rec_wheat": float(rec_wheat),
            "d_qk": clf.decision_function(K_va)
        }
        if best_res is None or f1 > best_res["f1"]:
            best_res = res
    return best_res


def softmax(d: np.ndarray) -> np.ndarray:
    z = d - d.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run_circuit_benchmark(alpha_fusion: float = 0.53) -> pd.DataFrame:
    tables, _, manifest, _ = load_data()
    train, val, F = tables["train"], tables["validation"], manifest["features"]
    X_tr_raw, X_va_raw = train[F].to_numpy(), val[F].to_numpy()
    y_tr, y_va = train.target.to_numpy(), val.target.to_numpy()

    prep = encoder().fit(X_tr_raw)
    X_tr_pca = prep.transform(X_tr_raw)
    X_va_pca = prep.transform(X_va_raw)

    results = []

    # 1. Baseline: Linear chain (a=0.2, g=0.01)
    def sv_linear(x, a=0.2, g=0.01):
        qc = QuantumCircuit(QUBITS)
        qc.h(range(QUBITS))
        for i in range(QUBITS):
            qc.rz(2 * a * x[i], i)
        for i in range(QUBITS - 1):
            qc.rzz(2 * g * x[i] * x[i + 1], i, i + 1)
        return Statevector.from_instruction(qc).data

    results.append(evaluate_kernel_circuit(
        lambda x: sv_linear(x, 0.2, 0.01), "Baseline (Linear chain, a=0.2, g=0.01)",
        X_tr_pca, X_va_pca, y_tr, y_va
    ))

    # 2. Qiskit ZZFeatureMap
    zz_map = ZZFeatureMap(feature_dimension=QUBITS, reps=1, entanglement="linear")
    results.append(evaluate_kernel_circuit(
        lambda x: Statevector.from_instruction(zz_map.assign_parameters(x)).data,
        "Qiskit ZZFeatureMap (reps=1, linear)",
        X_tr_pca, X_va_pca, y_tr, y_va
    ))

    # 3. Custom RY-RX + RZZ
    def sv_ry_rx(x, a=0.2, g=0.01):
        qc = QuantumCircuit(QUBITS)
        for i in range(QUBITS):
            qc.ry(a * x[i], i)
            qc.rx(a * x[i], i)
        for i in range(QUBITS - 1):
            qc.rzz(2 * g * x[i] * x[i + 1], i, i + 1)
        return Statevector.from_instruction(qc).data

    results.append(evaluate_kernel_circuit(
        sv_ry_rx, "Custom RY-RX + RZZ",
        X_tr_pca, X_va_pca, y_tr, y_va
    ))

    # 4. Data Re-uploading (2 layers + mixing)
    def sv_reupload(x, a=0.2, g=0.01):
        qc = QuantumCircuit(QUBITS)
        qc.h(range(QUBITS))
        for i in range(QUBITS):
            qc.rz(2 * a * x[i], i)
        for i in range(QUBITS - 1):
            qc.rzz(2 * g * x[i] * x[i + 1], i, i + 1)
        for i in range(QUBITS):
            qc.ry(np.pi / 4, i)
        for i in range(QUBITS):
            qc.rz(2 * a * x[i], i)
        for i in range(QUBITS - 1):
            qc.rzz(2 * g * x[i] * x[i + 1], i, i + 1)
        return Statevector.from_instruction(qc).data

    results.append(evaluate_kernel_circuit(
        sv_reupload, "Data Re-uploading (2 layers + RY(pi/4))",
        X_tr_pca, X_va_pca, y_tr, y_va
    ))

    # 5. Circular / Ring topology
    def sv_circ(x, a=0.2, g=0.01):
        qc = QuantumCircuit(QUBITS)
        qc.h(range(QUBITS))
        for i in range(QUBITS):
            qc.rz(2 * a * x[i], i)
        for i in range(QUBITS):
            qc.rzz(2 * g * x[i] * x[(i + 1) % QUBITS], i, (i + 1) % QUBITS)
        return Statevector.from_instruction(qc).data

    results.append(evaluate_kernel_circuit(
        sv_circ, "Circular / Ring Entanglement (8 pairs)",
        X_tr_pca, X_va_pca, y_tr, y_va
    ))

    # 6. Full All-to-all topology
    def sv_all(x, a=0.2, g=0.005):
        qc = QuantumCircuit(QUBITS)
        qc.h(range(QUBITS))
        for i in range(QUBITS):
            qc.rz(2 * a * x[i], i)
        for i in range(QUBITS):
            for j in range(i + 1, QUBITS):
                qc.rzz(2 * g * x[i] * x[j], i, j)
        return Statevector.from_instruction(qc).data

    results.append(evaluate_kernel_circuit(
        sv_all, "Full All-to-All Entanglement (28 pairs)",
        X_tr_pca, X_va_pca, y_tr, y_va
    ))

    # 7. No Entanglement (gamma=0)
    def sv_none(x, a=0.2):
        qc = QuantumCircuit(QUBITS)
        qc.h(range(QUBITS))
        for i in range(QUBITS):
            qc.rz(2 * a * x[i], i)
        return Statevector.from_instruction(qc).data

    results.append(evaluate_kernel_circuit(
        sv_none, "No Entanglement (Product State, gamma=0)",
        X_tr_pca, X_va_pca, y_tr, y_va
    ))

    # 8. Bandwidth variations
    for a_val in [0.05, 0.1, 0.3, 0.4]:
        results.append(evaluate_kernel_circuit(
            lambda x, a=a_val: sv_linear(x, a, 0.01), f"Bandwidth scaling a={a_val}",
            X_tr_pca, X_va_pca, y_tr, y_va
        ))

    # Evaluate fusion with classical RBF for each candidate
    model_rbf = MatchedRBF(gamma=0.11288, C=10.0).fit(X_tr_raw, y_tr)
    d_rbf = model_rbf.svm.decision_function(model_rbf.prep.transform(X_va_raw))
    P_rbf = softmax(d_rbf)

    fusion_rows = []
    for r in results:
        P_qk = softmax(r["d_qk"])
        P_hyb = alpha_fusion * P_qk + (1.0 - alpha_fusion) * P_rbf
        pred_h = model_rbf.svm.classes_[P_hyb.argmax(axis=1)]
        cm = confusion_matrix(y_va, pred_h, labels=LABELS)
        fusion_rows.append({
            "Architecture": r["name"],
            "QK_Macro_F1": r["f1"],
            "QK_Accuracy": r["acc"],
            "Hybrid_Macro_F1": float(f1_score(y_va, pred_h, average="macro")),
            "Hybrid_Accuracy": float(accuracy_score(y_va, pred_h)),
            "Hybrid_Correct": int((y_va == pred_h).sum()),
            "Wheat_Recall": float(cm[2, 2] / cm[2].sum()),
            "Maize_Recall": float(cm[1, 1] / cm[1].sum()),
            "Alfalfa_Recall": float(cm[0, 0] / cm[0].sum())
        })

    df = pd.DataFrame(fusion_rows)
    out_file = ROOT / "outputs/circuit_benchmark.json"
    df.to_json(out_file, indent=2, orient="records")
    return df


def main():
    print("Running quantum circuit design benchmark across 11 configurations...", flush=True)
    df = run_circuit_benchmark()
    print("\n================== BENCHMARK RÉSULTATS (outputs/circuit_benchmark.json) ==================")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
