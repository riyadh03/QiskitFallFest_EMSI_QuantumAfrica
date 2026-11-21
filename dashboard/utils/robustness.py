"""The perturbation experiments that actually exist, each with its exact protocol.

Three Gaussian-noise protocols were run at different times with different noise definitions. They are NOT
comparable with each other and are never drawn on the same chart.
"""
from . import data as D

PROTOCOLS = {
    'paired': dict(
        title='Gaussian noise, raw-feature scale, paired (5 draws)',
        short='Paired Gaussian noise',
        description='Validation features x are replaced by x + σ·s·ε, where s is the training standard deviation '
                    'of each raw band and ε ~ N(0, 1). Models are trained on clean data and never retrained. '
                    '5 noise draws per σ; every model sees exactly the same noisy copies (generator re-seeded '
                    'with 2026 for each model), so differences come from the models, not from the draws.',
        draws=5, has_sd=True, has_accuracy=True, source='export'),
    'reliability': dict(
        title='Gaussian noise, raw-feature scale (5 draws) + band drop',
        short='Gaussian noise + band drop',
        description='Same noise definition (σ × training std of each raw band, 5 draws, mean ± sd) applied to the '
                    'quantum kernel, the matched RBF and the supplied reference. Band drop replaces one of the 13 '
                    'features by its training mean, emulating a missing band (e.g. cloud-masked). Quantum kernel '
                    'predictions use svm.predict (clean score 0.940).',
        draws=5, has_sd=True, has_accuracy=False, source='reliability'),
    'pca_noise': dict(
        title='Gaussian noise in standardized PCA space (single draw)',
        short='PCA-space noise (single draw)',
        description='Notebook 03: N(0, σ) noise added to the standardized PCA-8 features (to the raw unscaled '
                    'features for Quantum Kitchen Sinks), one noise realization per σ. The fusion here used '
                    'α = 0.5, not the submitted 0.53. Because the noise is applied in a different space, the σ '
                    'values are not comparable with the raw-scale protocols.',
        draws=1, has_sd=False, has_accuracy=False, source='alternatives'),
}

NOT_IMPLEMENTED = ('Systematic sensor bias (constant offset or gain drift) and correlated multi-band dropout were '
                   'not implemented in this project.')


def series(protocol, metric='macro_f1'):
    """Return list of dict(key, x, y, sd) for one protocol; empty list when the source file is missing."""
    if protocol == 'paired':
        e = D.load('export')
        if not e:
            return []
        rows = e['robustness_paired']['rows']
        out = []
        for key in ['matched_rbf', 'quantum_kernel', 'hybrid_fusion']:
            r = [x for x in rows if x['model'] == key]
            out.append(dict(key=key, x=[x['sigma'] for x in r], y=[x[f'{metric}_mean'] for x in r],
                            sd=[x[f'{metric}_sd'] for x in r]))
        return out
    if protocol == 'reliability':
        r = D.load('reliability')
        if not r or metric != 'macro_f1':
            return []
        g = r['gaussian_noise']
        return [dict(key=k, x=[x['sigma'] for x in g], y=[x[k]['mean'] for x in g], sd=[x[k]['sd'] for x in g])
                for k in ['reference', 'matched_rbf', 'quantum_kernel']]
    if protocol == 'pca_noise':
        a = D.load('alternatives')
        if not a or metric != 'macro_f1':
            return []
        t = a['robustness_table']
        keymap = {'QKS': 'qks', 'Classical_RBF': 'matched_rbf', 'Quantum_Kernel': 'quantum_kernel',
                  'Fusion_Hybride': 'hybrid_fusion'}
        return [dict(key=v, x=t['sigma_levels'], y=t[k], sd=None,
                     label=(None if v != 'hybrid_fusion' else 'Decision fusion (α = 0.5 in this run)'))
                for k, v in keymap.items() if k in t]
    return []


def degradation(s):
    """Same series expressed as change relative to the clean (σ = 0) value."""
    return [dict(s_, y=[v - s_['y'][0] for v in s_['y']]) for s_ in s]


def band_drop():
    r = D.load('reliability')
    return r['feature_drop'] if r else None


def protocols_for(key):
    out = []
    for p in PROTOCOLS:
        if any(s['key'] == key for s in series(p)):
            out.append(p)
    return out
