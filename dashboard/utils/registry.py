"""The experiments actually run in this repository, in the order of the project story.

Numbers are read from the saved result files at runtime (utils.data); this module only adds the written context
(objective, method, interpretation). Fields a source does not record stay None and render as "Not recorded".
"""
from . import data as D

VAL = 'Validation · 150 fields (50 per class)'


def _metrics_from_run(run):
    if not run:
        return None
    m = run['metrics']
    return dict(macro_f1=m['macro_f1'], accuracy=m['accuracy'], precision=m['precision'], recall=m['recall'],
                confusion=m['confusion'])


def experiments():
    s = D.load('qk_summary') or {}
    env = (s.get('environment') or {})
    env_txt = f"Python {env['python']} · {env['machine']}" if env else None
    exp = D.load('export') or {}
    audit = D.load('audit') or {}
    ref_cfg = (D.load('reference_config') or {}).get('params', {})
    out = []

    # ------------------------------------------------------------------ classical
    st_ref = D.qk_seed_stats('reference')
    r = D.qk_run('reference')
    out.append(dict(
        key='reference', group='classical', phase='Phase 1 · Baseline',
        title='Supplied classical reference (RBF-SVM, 13 raw features)',
        objective='Establish the score any quantum or hybrid pipeline is compared against. This model is frozen by '
                  'the organizers and uses all 13 Sentinel-2 features.',
        method='StandardScaler on the 13 raw features (B1–B12, B8A, NDVI) → RBF support-vector machine. '
               'Hyper-parameters were fixed by the organizers using three spatial training folds.',
        config={'Model': 'RBF-SVM (scikit-learn SVC)', 'Features': '13 raw features, standardized',
                'C': ref_cfg.get('svc__C'), 'γ (RBF)': ref_cfg.get('svc__gamma'),
                'Seeds': '0, 1, 2 (deterministic)', 'Qubits': 0},
        metrics=_metrics_from_run(r), split=VAL,
        seeds_note='Deterministic: seeds 0, 1 and 2 give the identical result (sd = 0).',
        fit=st_ref and st_ref['fit'], predict=st_ref and st_ref['predict'],
        resources={'Support vectors': r and r['resources'].get('support_vectors'),
                   'SVM dual coefficients': r and r['resources'].get('svm_dual_coefficients')},
        findings=['Macro-F1 0.880 on validation; wheat is the weakest class (42/50, recall 84%).',
                  'Every PCA-8 model in this project beats it; the spatial-block bootstrap interval for '
                  'quantum kernel − reference excludes zero (+0.060, 95% CI [+0.026, +0.109]).'],
        strengths=['Fast (milliseconds), deterministic, no tuning on validation.'],
        weaknesses=['Wheat confused with alfalfa and maize (row [4, 4, 42]).'],
        limitations=['Frozen configuration; not tuned with the same effort as the project models.'],
        robustness=['reliability'],
        sources=[D.path('qk_summary'), D.path('reference_config'), D.path('reliability')],
        meta={'Experiment id': 'reference', 'Backend': 'CPU (classical)', 'Environment': env_txt}))

    st_rbf = D.qk_seed_stats('matched_rbf')
    r = D.qk_run('matched_rbf')
    cfg = s.get('matched_rbf_config', {})
    out.append(dict(
        key='matched_rbf', group='classical', phase='Phase 3 · Matched control',
        title='Matched classical control (RBF-SVM on PCA-8)',
        objective='Isolate the contribution of the quantum component: same input representation, same '
                  'preprocessing, same training rows and the same tuning effort as the quantum kernel, '
                  'with the quantum kernel replaced by a Gaussian (RBF) kernel.',
        method='StandardScaler → PCA(8) → StandardScaler (fitted on training rows only) → RBF-SVM. '
               'Tuned on an 80-configuration grid (20 γ × 4 C), the same size as the quantum-kernel grid.',
        config={'Model': 'RBF-SVM', 'Features': 'PCA(8), standardized', 'C': cfg.get('C'), 'γ (RBF)': cfg.get('gamma'),
                'Configurations tried': (s.get('tuning') or {}).get('rbf_configs'),
                'Tuning time': f"{(s.get('tuning') or {}).get('rbf_seconds', 0):.1f} s" if s else None,
                'Seeds': '0, 1, 2 (deterministic)', 'Qubits': 0},
        metrics=_metrics_from_run(r), split=VAL,
        seeds_note='Deterministic: seeds 0, 1 and 2 give the identical result (sd = 0).',
        fit=st_rbf and st_rbf['fit'], predict=st_rbf and st_rbf['predict'],
        resources={'Support vectors': r and r['resources'].get('support_vectors'),
                   'SVM dual coefficients': r and r['resources'].get('svm_dual_coefficients'),
                   'Encoder fitted values': r and r['resources'].get('encoder_fitted_values')},
        findings=['Macro-F1 0.932 — only 1 field behind the quantum kernel under the vote rule (140 vs 141).',
                  'Wheat recall stays at 84%: 6 wheat fields predicted as maize (row [2, 6, 42]).'],
        strengths=['Strongest maize recall of all models (100%). ~40× faster to fit than the quantum kernel (0.07 s vs 2.7 s).'],
        weaknesses=['Systematic wheat → maize confusion ("wheat bias").'],
        limitations=['Hyper-parameters selected on the same validation fields that report the score.'],
        robustness=['reliability', 'paired', 'pca_noise'],
        sources=[D.path('qk_summary'), D.path('grid_rbf'), D.path('export')],
        meta={'Experiment id': 'matched_rbf', 'Backend': 'CPU (classical)', 'Environment': env_txt}))

    mh = D.starter_model('matched_head')
    out.append(dict(
        key='matched_head', group='classical', phase='Phase 1 · Starter (supplied results)',
        title='Starter matched MLP control (organizer-supplied results)',
        objective='Parameter-matched classical control for the 4-qubit starter QNN, provided with the challenge.',
        method='PCA(4) → 4 → tanh(1 hidden unit) → softmax(3); trained with 100 COBYLA evaluations.',
        config={'Model': 'Tiny MLP', 'Features': 'PCA(4)', 'Hidden units': 1,
                'Trainable parameters': mh and mh['resources'].get('trainable'),
                'Optimizer budget': f"{mh['resources'].get('objective_evaluations')} COBYLA evaluations" if mh else None,
                'Seeds': '0, 1, 2 (mean ± sd)'},
        metrics=mh and dict(macro_f1=mh['macro_f1_mean'], accuracy=mh['accuracy_mean'], precision=mh['precision_mean'],
                            recall=mh['recall_mean'], confusion=mh['confusion_mean']),
        metrics_sd=mh and dict(macro_f1=mh['macro_f1_sd'], accuracy=mh['accuracy_sd']), split=VAL,
        seeds_note='Mean over seeds 0, 1, 2; the confusion matrix shows mean counts.',
        fit=mh and mh['fit_seconds_mean'], predict=mh and mh['predict_seconds_mean'],
        resources={'Trainable parameters': mh and mh['resources'].get('trainable')},
        findings=['Unstable across seeds (macro-F1 sd ≈ 0.11): 100 optimizer evaluations are not enough for '
                  'this control to converge.'],
        strengths=['Parameter count close to the QNN (11 vs 8).'],
        weaknesses=['Collapses on alfalfa for some seeds.'],
        limitations=['Supplied by the organizers; not re-run by the team.'],
        robustness=[], sources=[D.path('starter')],
        meta={'Experiment id': 'matched_head (results/)', 'Backend': 'CPU (classical)'}))

    tw = ((audit.get('validation_scores') or {}).get('reported_selected_on_validation') or {}).get('classical_twin_kernel')
    out.append(dict(
        key='classical_twin', group='classical', phase='Audit · Is the gain quantum?',
        title='Classical twin of the quantum kernel (closed form, no circuit)',
        objective='Test whether the quantum kernel needs a quantum circuit at all. Without entanglement the circuit '
                  'kernel equals K(x, x′) = ∏ᵢ cos²(a·(xᵢ − x′ᵢ)), computable directly on a laptop.',
        method='Same PCA-8 encoder and SVM; the fidelity kernel is replaced by the closed-form product of cosines '
               '(a = 0.2, C = 100). Also fused with the same RBF at α = 0.53.',
        config={'Model': 'SVM, precomputed classical kernel', 'Kernel': '∏ cos²(a Δx)', 'a': 0.2, 'C': 100,
                'Seeds': 'Deterministic', 'Qubits': 0},
        metrics=tw and dict(macro_f1=tw['macro_f1'], accuracy=tw['accuracy'], precision=None,
                            recall=[None, None, tw['recall_wheat']], confusion=None),
        split=VAL, seeds_note='Deterministic.', fit=None, predict=None,
        resources={'Max error vs γ=0 circuit': f"{s.get('closed_form_error', 0):.1e}" if s else None},
        findings=['Macro-F1 0.940 (141/150) and 90% wheat recall, versus 0.947 and 92% for the circuit kernel.',
                  'Fused with the RBF it reaches 142/150 — one field fewer than the submitted fusion.'],
        strengths=['Shows that the kernel shape, not entanglement, explains most of the wheat improvement.'],
        weaknesses=['Per-class precision and confusion matrix not exported by the audit.'],
        limitations=['Single validation split; differences of 1 field are within bootstrap noise.'],
        robustness=[], sources=[D.path('audit')],
        meta={'Experiment id': 'classical_twin_kernel (audit)', 'Backend': 'CPU (classical)'}))

    # ------------------------------------------------------------------ quantum
    sq = D.starter_model('qnn')
    out.append(dict(
        key='starter_qnn', group='quantum', phase='Phase 1 · Starter (supplied results)',
        title='Starter variational QNN (4 qubits, organizer-supplied results)',
        objective='Diagnose the supplied quantum starting point before designing our own model.',
        method='PCA(4) → angles (π/2)(1 + tanh x) → 4 RY encoding gates → real_amplitudes ansatz (1 repetition, '
               'linear CX) → parity readout grouped into 3 classes; trained with 100 COBYLA evaluations.',
        config={'Model': 'Variational QNN', 'Features': 'PCA(4)', 'Encoding': 'RY angle encoding',
                'Qubits': sq and sq['resources'].get('qubits'), 'Circuit depth': sq and sq['resources'].get('depth'),
                'Two-qubit gates': sq and sq['resources'].get('two_qubit_gates'),
                'Trainable angles': sq and sq['resources'].get('trainable'),
                'Optimizer budget': '100 COBYLA evaluations', 'Seeds': '0, 1, 2 (mean ± sd)'},
        metrics=sq and dict(macro_f1=sq['macro_f1_mean'], accuracy=sq['accuracy_mean'], precision=sq['precision_mean'],
                            recall=sq['recall_mean'], confusion=sq['confusion_mean']),
        metrics_sd=sq and dict(macro_f1=sq['macro_f1_sd'], accuracy=sq['accuracy_sd']), split=VAL,
        seeds_note='Mean over seeds 0, 1, 2; the confusion matrix shows mean counts.',
        fit=sq and sq['fit_seconds_mean'], predict=sq and sq['predict_seconds_mean'],
        resources={'Trainable angles': sq and sq['resources'].get('trainable'),
                   'Objective evaluations': sq and sq['resources'].get('objective_evaluations'),
                   'Compilation': sq and sq['resources'].get('compilation')},
        findings=['Macro-F1 0.856 ± 0.006 — below the supplied classical reference (0.880).',
                  'Training takes ~55 s per seed for 8 angles: the optimizer budget, not the data, limits it.'],
        strengths=['Stable across seeds; beats its tiny MLP control.'],
        weaknesses=['Shallow circuit with 4 features loses information that PCA(8) keeps.'],
        limitations=['Supplied by the organizers; motivated our move to an 8-qubit kernel method.'],
        robustness=[], sources=[D.path('starter')],
        meta={'Experiment id': 'qnn (results/)', 'Backend': 'Exact statevector simulation (CPU)'}))

    qb = D.alternative('QNN')
    out.append(dict(
        key='qnn_benchmark', group='quantum', phase='Phase 4 · Architecture benchmark',
        title='Variational QNN, L=1 (team benchmark run)',
        objective='Re-run the variational approach inside the 5-architecture benchmark (docs/HINTS.md).',
        method='Same QuantumClassifier as the starter (RY encoding, L = 1, 100 COBYLA evaluations). '
               'For speed it was fitted on the first 200 training rows only (notebook 03, cell 4).',
        config={'Model': 'Variational QNN', 'Features': qb and qb['features'], 'Qubits': qb and qb['qubits'],
                'Trainable angles': qb and qb['params'], 'Training rows': '200 (subset, not 914)', 'Seed': 0},
        metrics=qb and dict(macro_f1=qb['macro_f1'], accuracy=qb['accuracy'], precision=None,
                            recall=[None, None, qb['wheat_recall']], confusion=None),
        split=VAL, seeds_note='Single run, seed 0.', fit=qb and qb['train_time_s'], predict=None,
        resources={'Trainable angles': qb and qb['params']},
        findings=['Macro-F1 0.806 — the weakest architecture in the benchmark.'],
        strengths=['Genuine trainable quantum circuit.'],
        weaknesses=['Slowest to train (≈60 s) for the lowest score.'],
        limitations=['Trained on 200 of 914 rows, so it is not a like-for-like comparison with the other '
                     'architectures. No confusion matrix or noise test was recorded for this run.'],
        robustness=[], sources=[D.path('alternatives'), 'notebooks/03_alternatives_benchmark.ipynb'],
        meta={'Experiment id': 'alternatives · QNN', 'Backend': 'Exact statevector simulation (CPU)'}))

    qks = D.alternative('Quantum Kitchen')
    out.append(dict(
        key='qks', group='quantum', phase='Phase 4 · Architecture benchmark',
        title='Quantum Kitchen Sinks (60 random features)',
        objective='Test a cheap hybrid alternative: random quantum feature maps followed by a linear classifier '
                  '(Wilson et al., 2018).',
        method='15 random 4-qubit maps on the 13 raw features → 60 features → logistic regression (C = 1). '
               'The circuit expectation values are computed with a closed-form expression; no circuit is simulated.',
        config={'Model': 'Random features + logistic regression', 'Features': qks and qks['features'],
                'Qubits per map': qks and qks['qubits'], 'Maps': 15, 'Random seed': 2026,
                'Trainable parameters': qks and qks['params']},
        metrics=qks and dict(macro_f1=qks['macro_f1'], accuracy=qks['accuracy'], precision=None,
                             recall=[None, None, qks['wheat_recall']], confusion=None),
        split=VAL, seeds_note='Single run (feature seed 2026).', fit=qks and qks['train_time_s'], predict=None,
        resources={'Trainable parameters': qks and qks['params']},
        findings=['Macro-F1 0.827: fast (≈50 ms) but far below the kernel methods.'],
        strengths=['Training in milliseconds.'],
        weaknesses=['Random features on unscaled raw bands; wheat recall 82%.'],
        limitations=['Expectation values computed analytically, so no quantum simulation is involved. '
                     'Single random draw; no confusion matrix recorded.'],
        robustness=['pca_noise'], sources=[D.path('alternatives'), 'notebooks/03_alternatives_benchmark.ipynb'],
        meta={'Experiment id': 'alternatives · QKS', 'Backend': 'CPU (closed-form features)'}))

    st_qk = D.qk_seed_stats('quantum_kernel')
    r = D.qk_run('quantum_kernel')
    fc = s.get('final_config', {})
    rules = exp.get('quantum_kernel_decision_rules', {})
    out.append(dict(
        key='quantum_kernel', group='quantum', phase='Phase 3 · Main quantum model',
        title='8-qubit fidelity quantum kernel SVM',
        objective='Replace the hard-to-train QNN by a kernel method: map each field to an 8-qubit state and let an '
                  'SVM learn from state overlaps K(x, x′) = |⟨ψ(x)|ψ(x′)⟩|².',
        method='StandardScaler → PCA(8) → StandardScaler → feature map H · RZ(2a·xᵢ) · RZZ(2γ·xᵢxᵢ₊₁) on a linear '
               'chain → exact statevectors → fidelity kernel → SVM with precomputed kernel. No trainable angles.',
        config={'Model': 'SVM, precomputed fidelity kernel', 'Features': 'PCA(8), standardized',
                'Qubits': r and r['resources']['qubits'], 'a (angle scale)': fc.get('a'), 'γ (RZZ strength)': fc.get('gamma'),
                'Repetitions': fc.get('reps'), 'C': fc.get('C'), 'Entanglement': 'Linear chain, 7 RZZ',
                'Configurations tried': (s.get('tuning') or {}).get('quantum_configs'),
                'Tuning time': f"{(s.get('tuning') or {}).get('quantum_seconds', 0):.1f} s" if s else None,
                'Seeds': '0, 1, 2 (deterministic)'},
        metrics=_metrics_from_run(r), split=VAL,
        seeds_note='Deterministic: seeds 0, 1 and 2 give the identical result (sd = 0).',
        fit=st_qk and st_qk['fit'], predict=st_qk and st_qk['predict'],
        resources={'Logical depth': r and r['resources'].get('logical_depth'),
                   'RZZ gates': r and r['resources'].get('rzz_gates'),
                   'Support vectors': r and r['resources'].get('support_vectors'),
                   'Statevector simulations': r and r['resources'].get('statevector_simulations'),
                   'Kernel matrix floats': r and r['resources'].get('kernel_matrix_floats')},
        findings=['Macro-F1 0.940 with svm.predict (141/150); 0.947 (142/150) when classes are chosen by the largest '
                  'decision margin — field 56 is a perfect 3-way vote tie.' if rules else
                  'Macro-F1 0.940 on validation.',
                  'Best wheat recall of all models (92%, row [1, 3, 46]). Against the matched RBF this is the only '
                  'difference whose spatial-block bootstrap interval excludes zero (+8 pts, CI [+1.6, +20]).'],
        strengths=['No trainable circuit parameters, so no barren plateaus; deterministic; fits in ~3 s.',
                   'Most robust model under strong Gaussian noise (σ = 0.5) in the paired test.'],
        weaknesses=['Overall macro-F1 is not significantly different from the matched RBF.'],
        limitations=['Removing entanglement (γ = 0) gives the same score: the gain is not attributable to '
                     'entanglement. Hyper-parameters were selected on validation.'],
        robustness=['reliability', 'paired', 'pca_noise'],
        sources=[D.path('qk_summary'), D.path('final_config'), D.path('grid_quantum'), D.path('export')],
        meta={'Experiment id': 'quantum_kernel', 'Backend': 'Exact statevector simulation (Qiskit, CPU)',
              'IBM hardware': 'Kernel rows also measured on ibm_kingston (see Hardware page)', 'Environment': env_txt}))

    st_ne = D.qk_seed_stats('no_entanglement')
    r = D.qk_run('no_entanglement')
    ne = s.get('no_entanglement_config', {})
    out.append(dict(
        key='no_entanglement', group='quantum', phase='Phase 3 · Ablation',
        title='Ablation: quantum kernel without entanglement (γ = 0)',
        objective='Answer the focused question: does the entangling part of the feature map add anything?',
        method='Same pipeline as the 8-qubit kernel with all RZZ gates removed (product state). Tuned on its own '
               'grid (5 a × 4 C) with the same selection rule.',
        config={'Model': 'SVM, precomputed fidelity kernel', 'Features': 'PCA(8)', 'Qubits': 8, 'a': ne.get('a'),
                'γ': 0, 'C': ne.get('C'), 'Configurations tried': 20, 'Seeds': '0, 1, 2 (deterministic)'},
        metrics=_metrics_from_run(r), split=VAL,
        seeds_note='Deterministic: seeds 0, 1 and 2 give the identical result (sd = 0).',
        fit=st_ne and st_ne['fit'], predict=st_ne and st_ne['predict'],
        resources={'Logical depth': r and r['resources'].get('logical_depth'), 'RZZ gates': 0,
                   'Support vectors': r and r['resources'].get('support_vectors')},
        findings=['Identical accuracy to the entangled kernel (0.94); the two differ on 2 fields only.',
                  f"The kernel equals the closed form ∏cos²(aΔx) to {s.get('closed_form_error', 0):.1e}." if s else ''],
        strengths=['Depth 2 instead of 9, no two-qubit gates — trivially simulable classically.'],
        weaknesses=['—'],
        limitations=['A clear negative finding: on this data, entanglement adds nothing measurable.'],
        robustness=[], sources=[D.path('qk_summary')],
        meta={'Experiment id': 'no_entanglement', 'Backend': 'Exact statevector simulation (CPU)',
              'Environment': env_txt}))

    # ------------------------------------------------------------------ hybrid
    fu = exp.get('fusion') or {}
    fm = fu.get('metrics')
    tcv = ((audit.get('validation_scores') or {}).get('selected_on_train_cv_only') or {}).get('fusion_qk_rbf')
    out.append(dict(
        key='hybrid_fusion', group='hybrid', phase='Phase 4 · Submitted model',
        title='Calibrated decision fusion — submitted model',
        objective='Combine the RBF\'s confidence on maize with the quantum kernel\'s precision on wheat '
                  '(Grossi et al., 2022).',
        method='Both SVMs share the PCA-8 encoder. Their decision margins are turned into probabilities with a '
               'softmax and blended: P = α·softmax(d_QK) + (1 − α)·softmax(d_RBF); the class with the largest P '
               'wins. α was chosen by a 101-point scan on validation.',
        config={'Model': 'Decision-level fusion of two SVMs', 'Features': 'PCA(8), shared',
                'α (quantum weight)': fu.get('alpha'),
                'QK branch': 'a = {a}, γ = {gamma}, reps = {reps}, C = {C}'.format(**fu['config']['qk']) if fu else None,
                'RBF branch': 'γ = {gamma}, C = {C}'.format(**fu['config']['rbf']) if fu else None,
                'Qubits': 8, 'Seeds': '0, 1, 2 (deterministic)'},
        metrics=fm and dict(macro_f1=fm['macro_f1'], accuracy=fm['accuracy'], precision=fm['precision'],
                            recall=fm['recall'], confusion=fm['confusion']),
        split=VAL, seeds_note='Deterministic: seeds 0, 1 and 2 give the identical result (sd = 0).',
        fit=fu.get('fit_seconds'), predict=fu.get('predict_150_seconds'),
        resources={'Branches': 'Quantum kernel SVM + RBF-SVM', 'Fusion parameters': '1 (α)'},
        findings=['Macro-F1 0.953 (143/150) on validation, with ≥ 92% recall on every crop.',
                  'Only 1 field better than the quantum kernel alone (argmax rule) and the 143 plateau spans only '
                  'α ∈ [0.52, 0.54].',
                  f"With every choice made by cross-validation on training rows only, it scores {tcv['macro_f1']:.3f} "
                  f"({tcv['correct']}/150): 0.953 is an optimistic estimate." if tcv else ''],
        strengths=['Balanced per-class recall; deterministic; full train + predict in ~2 s.'],
        weaknesses=['Gain over each branch is within bootstrap noise.'],
        limitations=['α and both branches were tuned on validation. On IBM hardware the fusion never beats the '
                     'RBF alone (see Hardware page).'],
        robustness=['paired', 'pca_noise'],
        sources=[D.path('export'), D.path('audit'), 'challenge/submission.py'],
        meta={'Experiment id': 'hybrid_fusion (challenge/submission.py)',
              'Backend': 'Exact statevector simulation (QK) + CPU (RBF)',
              'Export generated (UTC)': exp.get('generated_utc'),
              'Environment': (f"Python {exp['environment']['python']} · Qiskit {exp['environment']['qiskit']}"
                              if exp.get('environment') else None),
              'IBM hardware': 'Quantum branch replayed on ibm_kingston kernel rows (see Hardware page)'}))
    return out


GROUPS = [('classical', 'Classical baseline & controls'), ('quantum', 'Quantum approaches'),
          ('hybrid', 'Hybrid decision fusion')]
