"""Paired per-item statistics for the method lineage (VolETA, VolTex, VolETA++).

Inputs are transcribed from the publications; outputs feed notes/facts-stats-lineage.md.
Run: python3 notes/stats_lineage.py   (standard library only; deterministic)

Sources
- VolETA and VolTex: P3 Table 1 (PDF p. 6), per-item APE (%) and Chamfer distance with
  transforms, transcribed in notes/facts-p3-voltex.md. The VolETA column matches P5 Table IV.
- VolETA++: P4 supplementary per-scene backbone table (tab:results_one_shot_mtf_easy_meduim,
  NeuS2 columns). The main paper prints only means. This table has anomalies (Appendix B):
  its NeuS2 mean APE is 4.156 against 4.26 printed in the main paper, and item 1 repeats
  VolETA's published result. Results involving VolETA++ are therefore reported with that caveat,
  and also without item 1.
"""
import itertools
import random

ITEMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
APE = {
    'VolETA':   [3.97, 22.64, 11.7, 5.46, 0.81, 6.02, 1.13, 7.79, 3.72, 5.8, 5.61, 18.07, 9.22],
    'VolTex':   [3.96, 14.16, 7.65, 4.48, 2.72, 1.55, 2.24, 3.24, 10.82, 1.19, 2.67, 2.05, 1.29],
    'VolETA++': [3.971, 0.671, 1.607, 8.611, 6.359, 2.704, 7.814, 4.418, 1.337, 4.201, 2.137, 1.933, 8.267],
}
CD = {
    'VolETA':   [0.0016, 0.0071, 0.0137, 0.0020, 0.0137, 0.0067, 0.0047, 0.0030, 0.0039, 0.0027, 0.0034, 0.0052, 0.0043],
    'VolTex':   [0.0015, 0.0051, 0.0144, 0.0025, 0.0114, 0.0054, 0.0054, 0.0030, 0.0059, 0.0025, 0.0041, 0.0042, 0.0047],
    'VolETA++': [0.0016, 0.0061, 0.0046, 0.0015, 0.0025, 0.0017, 0.0027, 0.0019, 0.0026, 0.0021, 0.0033, 0.0012, 0.0041],
}


def mean(x):
    return sum(x) / len(x)


def ranks(values):
    """Average ranks (1-based) with ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


# Differences are rounded to the precision of the coarser printed column, so that equal
# printed values count as ties (APE: 2 decimals; Chamfer distance: 4 decimals).
DIGITS = {'APE': 2, 'Chamfer': 4}


def wilcoxon_exact(a, b, digits=10):
    """Two-sided exact Wilcoxon signed-rank test of a - b (mid-ranks for ties, zero
    differences dropped). Returns (n_used, W_plus, p)."""
    d = [round(x - y, digits) for x, y in zip(a, b)]
    d = [v for v in d if v != 0]
    r = ranks([abs(v) for v in d])
    w_plus = sum(rk for rk, v in zip(r, d) if v > 0)
    total = sum(r)
    centre = total / 2
    obs = abs(w_plus - centre)
    hits = 0
    for signs in itertools.product((0, 1), repeat=len(r)):
        w = sum(rk for rk, s in zip(r, signs) if s)
        if abs(w - centre) >= obs - 1e-12:
            hits += 1
    return len(d), w_plus, hits / 2 ** len(r)


def bootstrap_ci(x, b=10000, seed=0):
    rng = random.Random(seed)
    n = len(x)
    means = sorted(mean([x[rng.randrange(n)] for _ in range(n)]) for _ in range(b))
    return means[int(0.025 * b)], means[int(0.975 * b) - 1]


def wins(new, old, digits=10):
    """Items on which `new` has the lower error, and ties."""
    d = [round(x - y, digits) for x, y in zip(new, old)]
    return sum(1 for v in d if v < 0), sum(1 for v in d if v == 0)


def drop(x, items=(1,)):
    """Remove the given item IDs (the anomalous items 1 and 11 of the supplementary table)."""
    return [v for i, v in zip(ITEMS, x) if i not in items]


def report():
    out = []
    for m in APE:
        lo, hi = bootstrap_ci(APE[m])
        out.append(f'MAPE {m}: mean {mean(APE[m]):.2f}, bootstrap 95% CI [{lo:.2f}, {hi:.2f}]')
    pairs = [('VolTex', 'VolETA'), ('VolETA++', 'VolTex'), ('VolETA++', 'VolETA')]
    for metric, data in (('APE', APE), ('Chamfer', CD)):
        k = DIGITS[metric]
        for new, old in pairs:
            low, ties = wins(data[new], data[old], k)
            n, w, p = wilcoxon_exact(data[new], data[old], k)
            line = f'{metric} {new} vs {old}: {new} lower on {low} of 13 (ties {ties}); Wilcoxon n={n}, W+={w:g}, p={p:.4f}'
            if 'VolETA++' in (new, old):
                p1 = wilcoxon_exact(drop(data[new]), drop(data[old]), k)[2]
                p2 = wilcoxon_exact(drop(data[new], (1, 11)), drop(data[old], (1, 11)), k)[2]
                line += f'; without item 1: p={p1:.4f}; without items 1 and 11: p={p2:.4f}'
            out.append(line)
    return out


# Per-item results of the two other challenge methods, P3 Table 1 (ININ. = HR-CMS, FoodR. = GS-GP),
# transcribed in notes/facts-p3-voltex.md; used only for the rank correlation of volume and shape.
APE_OTHER = {
    'HR-CMS': [2.28, 16.08, 89.63, 0.27, 9.91, 8.96, 1.97, 0.47, 8.71, 3.46, 10.24, 20.01, 16.06],
    'GS-GP':  [15.52, 14.59, 34.63, 17.76, 0.84, 9.39, 11.86, 4.67, 0.45, 1.94, 0.96, 8.57, 12.22],
}
CD_OTHER = {
    'HR-CMS': [0.002, 0.0036, 0.0049, 0.0038, 0.002, 0.0038, 0.0048, 0.0019, 0.0029, 0.0034, 0.0015, 0.0026, 0.0044],
    'GS-GP':  [0.0011, 0.0031, 0.0053, 0.0015, 0.0040, 0.0025, 0.0025, 0.0010, 0.0033, 0.0019, 0.0015, 0.0041, 0.0046],
}


def paired_bootstrap_ci(new, old, b=10000, seed=0):
    """Percentile 95% interval of the mean paired difference new - old, resampling items."""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(new, old)]
    n = len(d)
    means = sorted(mean([d[rng.randrange(n)] for _ in range(n)]) for _ in range(b))
    return mean(d), means[int(0.025 * b)], means[int(0.975 * b) - 1]


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def spearman_perm_p(x, y, b=10000, seed=0):
    """Two-sided permutation p-value of Spearman's rho (shuffling y)."""
    rng = random.Random(seed)
    r0 = abs(spearman(x, y))
    y2 = list(y)
    hits = 0
    for _ in range(b):
        rng.shuffle(y2)
        if abs(spearman(x, y2)) >= r0 - 1e-12:
            hits += 1
    return (hits + 1) / (b + 1)


T975_12 = 2.178813  # 97.5% quantile of Student's t with 12 degrees of freedom (13 items)


def t_ci(x):
    """Mean and 95% t interval (n = 13). Percentile bootstrap intervals are too narrow with so few items."""
    assert len(x) == 13
    m = mean(x); half = T975_12 * sd(x) / len(x) ** 0.5
    return m, m - half, m + half


def extra():
    out = []
    for name in APE:
        m, lo, hi = t_ci(APE[name])
        out.append(f'ST-1t MAPE {name}: {m:.2f}, 95% t interval [{lo:.2f}, {hi:.2f}]')
    for new, old, tag in (('VolETA++', 'VolTex', 'ST-9t'), ('VolETA++', 'VolETA', 'ST-9bt')):
        m, lo, hi = t_ci([a - b for a, b in zip(APE[new], APE[old])])
        out.append(f'{tag} APE {new} minus {old}: mean {m:.2f} points, paired 95% t interval [{lo:.2f}, {hi:.2f}]')
    m, lo, hi = paired_bootstrap_ci(APE['VolETA++'], APE['VolTex'])
    out.append(f'ST-9 APE VolETA++ minus VolTex: mean {m:.2f} points, paired bootstrap 95% CI [{lo:.2f}, {hi:.2f}]')
    m, lo, hi = paired_bootstrap_ci(APE['VolETA++'], APE['VolETA'])
    out.append(f'ST-9b APE VolETA++ minus VolETA: mean {m:.2f} points, paired bootstrap 95% CI [{lo:.2f}, {hi:.2f}]')
    xs, ys = [], []
    for name in ('VolETA', 'VolTex'):
        r = spearman(APE[name], CD[name]); p = spearman_perm_p(APE[name], CD[name])
        out.append(f'ST-10 {name}: Spearman rho(APE, Chamfer) over 13 items = {r:.2f}, permutation p = {p:.2f}')
        xs += APE[name]; ys += CD[name]
    for name in APE_OTHER:
        r = spearman(APE_OTHER[name], CD_OTHER[name]); p = spearman_perm_p(APE_OTHER[name], CD_OTHER[name])
        out.append(f'ST-10 {name}: Spearman rho(APE, Chamfer) over 13 items = {r:.2f}, permutation p = {p:.2f}')
        xs += APE_OTHER[name]; ys += CD_OTHER[name]
    # A pooled rho over the 52 pairs would mix the between-method association with the
    # within-method one and treat repeated items as independent; use a stratified test instead.
    apes = [APE['VolETA'], APE['VolTex'], APE_OTHER['HR-CMS'], APE_OTHER['GS-GP']]
    cds = [CD['VolETA'], CD['VolTex'], CD_OTHER['HR-CMS'], CD_OTHER['GS-GP']]
    r, p = stratified_spearman(apes, cds)
    out.append(f'ST-10 mean within-method rho over the four methods = {r:.2f}, permutation within methods p = {p:.2f}')
    lo, hi = fisher_ci(spearman(APE['VolETA'], CD['VolETA']))
    out.append(f'ST-10 VolETA rho approximate 95% CI (Fisher z, 1/sqrt(n-3)) [{lo:.2f}, {hi:.2f}]')
    size = [spearman(c, GT13) for c in cds]
    out.append('ST-10 rho(Chamfer, reference volume) per method: ' + ', '.join(f'{v:.2f}' for v in size))
    norm = [spearman(a, [c / g ** (2 / 3) for c, g in zip(c, GT13)]) for a, c in zip(apes, cds)]
    out.append('ST-10 rho(APE, Chamfer / V^(2/3)) per method: ' + ', '.join(f'{v:.2f}' for v in norm))
    out.append(f'ST-11 scale accuracy implied by a 4.26% volume error if all of it came from scale: {((1 + 0.0426) ** (1 / 3) - 1) * 100:.2f}%')
    out += signed()
    # ST-14: paired tests among the challenge methods (P3 Table 1 columns) and of VolETA++ against them
    for a, b, A, B in (('VolETA', 'GS-GP', APE['VolETA'], APE_OTHER['GS-GP']), ('VolETA', 'HR-CMS', APE['VolETA'], APE_OTHER['HR-CMS']),
                       ('VolETA++', 'GS-GP', APE['VolETA++'], APE_OTHER['GS-GP']), ('VolETA++', 'HR-CMS', APE['VolETA++'], APE_OTHER['HR-CMS'])):
        low, ties = wins(A, B, 2)
        n, w, p = wilcoxon_exact(A, B, 2)
        out.append(f'ST-14 APE {a} vs {b}: {a} lower on {low} of 13 (ties {ties}); exact Wilcoxon p = {p:.3f}')
    return out


# P5 Table IV (PDF p. 11; source sec/5_experiments.tex): predicted and reference (GT) volumes in
# cm^3 of the 13 multi-view items, for the three challenge methods. P3 and P4 print only absolute
# errors, so signed errors exist only for these three.
GT13 = [38.53, 280.36, 249.65, 295.13, 392.58, 218.31, 368.77, 173.13, 232.74, 163.23, 85.18, 308.28, 589.82]
PRED = {
    'VolETA': [40.06, 216.90, 278.86, 279.02, 395.76, 205.17, 372.93, 186.62, 224.08, 153.76, 80.40, 363.99, 535.44],
    'HR-CMS': [37.65, 325.44, 473.40, 294.32, 353.66, 237.88, 361.49, 172.32, 253.01, 157.58, 76.46, 246.60, 495.10],
    'GS-GP':  [44.51, 321.26, 336.11, 347.54, 389.28, 197.82, 412.52, 181.21, 233.79, 160.06, 86.00, 334.70, 517.75],
}


def sd(x):
    m = mean(x)
    return (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5


def fisher_ci(r, n=13):
    import math
    z, h = math.atanh(r), 1.96 / (n - 3) ** 0.5
    return math.tanh(z - h), math.tanh(z + h)


def stratified_spearman(xs, ys, b=10000, seed=0):
    """Mean within-group Spearman rho and its permutation p-value, shuffling within groups."""
    rng = random.Random(seed)
    r0 = mean([spearman(x, y) for x, y in zip(xs, ys)])
    hits = 0
    for _ in range(b):
        r = []
        for x, y in zip(xs, ys):
            y2 = list(y); rng.shuffle(y2); r.append(spearman(x, y2))
        if abs(mean(r)) >= abs(r0) - 1e-12:
            hits += 1
    return r0, (hits + 1) / (b + 1)


def signed():
    out = []
    for m, pred in PRED.items():
        e = [100 * (p - g) / g for p, g in zip(pred, GT13)]
        over = sum(1 for v in e if v > 0)
        p = wilcoxon_exact(e, [0] * 13, 2)[2]
        lo, hi = bootstrap_ci(e)
        out.append(f'ST-12 {m} signed error over 13 items: mean {mean(e):+.2f}%, SD {sd(e):.2f}, '
                   f'median {sorted(e)[6]:+.2f}, over {over} under {13 - over}, Wilcoxon p = {p:.2f}, '
                   f'bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}], 95% limits of agreement '
                   f'[{mean(e) - 1.96 * sd(e):+.1f}, {mean(e) + 1.96 * sd(e):+.1f}]')
        _, tlo, thi = t_ci(e)
        out.append(f'ST-12t {m} signed error: 95% t interval [{tlo:+.2f}, {thi:+.2f}]')
    t = 100 * (sum(PRED['VolETA']) - sum(GT13)) / sum(GT13)
    out.append(f'ST-12 VolETA total volume of the 13 items: {t:+.2f}%')
    for m, a in list(APE.items()) + list(APE_OTHER.items()):
        rms = mean([v * v for v in a]) ** 0.5
        out.append(f'ST-13 {m}: items within 10% = {sum(1 for v in a if v < 10)} of 13, '
                   f'largest APE {max(a):.2f}, RMS of APE {rms:.2f}')
    return out


if __name__ == '__main__':
    # Self-check: the transcribed columns reproduce the printed means.
    assert round(mean(APE['VolETA']), 2) == 7.84
    assert round(mean(APE['VolTex']), 2) == 4.46
    assert round(mean(APE['VolETA++']), 3) == 4.156
    assert round(mean(CD['VolETA']), 4) == 0.0055
    assert round(mean(CD['VolTex']), 4) == 0.0054
    assert round(mean(CD['VolETA++']), 5) == 0.00276
    # Self-check of the exact test on a case with a known answer: all 5 differences positive.
    assert abs(wilcoxon_exact([2, 3, 4, 5, 6], [1, 1, 1, 1, 1])[2] - 2 / 32) < 1e-12
    assert round(mean(APE_OTHER['HR-CMS']), 2) == 14.47 and round(mean(APE_OTHER['GS-GP']), 2) == 10.26
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    assert abs(t_ci([1.0] * 12 + [14.0])[0] - 2.0) < 1e-12  # mean of the t interval
    # Self-check: the signed errors reproduce the printed APE of P5 Table IV.
    for m, a in (('VolETA', APE['VolETA']), ('HR-CMS', APE_OTHER['HR-CMS']), ('GS-GP', APE_OTHER['GS-GP'])):
        assert [round(abs(100 * (p - g) / g), 2) for p, g in zip(PRED[m], GT13)] == a, m
    print('\n'.join(report() + extra()))
