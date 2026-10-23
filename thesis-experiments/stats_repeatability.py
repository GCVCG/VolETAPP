"""Derived statistics of the unpublished experiments of the thesis (Section 9.8; ledger IDs UR-*).

Inputs (this folder): per-run volumes of NeuS2 reconstructions in NeuS2 units (closed pieces after the 5%
clean-up), five runs per item; regenerated checkerboard scales and unit factors. Standard library only.
Run from this folder: python3 stats_repeatability.py
"""
import csv, itertools, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # stats_lineage.py beside this file (release) or one level up (thesis notes)
from stats_lineage import wilcoxon_exact  # noqa: E402  (exact signed-rank test of the thesis)
ITEMS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "13", "14"]
GT = {"1": 38.53, "2": 280.36, "3": 249.65, "4": 295.13, "5": 392.58, "6": 218.31, "7": 368.77, "8": 173.13,
      "9": 232.74, "10": 163.23, "11": 85.18, "13": 308.28, "14": 589.82}  # P3 Table 1
APE_VOLETAPP = [3.971, 0.671, 1.607, 8.611, 6.359, 2.704, 7.814, 4.418, 1.337, 4.201, 2.137, 1.933, 8.267]  # P4 suppl.
APE_VOLTEX = [3.96, 14.16, 7.65, 4.48, 2.72, 1.55, 2.24, 3.24, 10.82, 1.19, 2.67, 2.05, 1.29]  # P3 Table 1
TABLE4_MEAN, TABLE4_SD = 0.1055, 0.0137  # P4-R14


def runs(name):
    out = {}
    for r in csv.DictReader(open(os.path.join(HERE, name))):
        out.setdefault(r["item"], []).append((r["seed"], float(r["volume_raw"])))
    return out


def cv(v):
    return 100 * st.stdev(v) / st.mean(v)


def summary(name):
    data = runs(name)
    failed = [(i, s) for i in data for s, v in data[i] if v == 0]
    cvs = {i: cv([v for _, v in data[i]]) for i in ITEMS}
    ok = {i: cv([v for _, v in data[i] if v > 0]) for i in ITEMS}
    return data, cvs, ok, failed


def same_seed(a, b):
    """Relative difference of the same (item, seed) on two GPUs, in % of the first."""
    A, B = runs(a), runs(b)
    d = []
    for i in ITEMS:
        sa, sb = dict(A.get(i, [])), dict(B.get(i, []))
        d += [100 * abs(sb[s] - sa[s]) / sa[s] for s in sa.keys() & sb.keys() if sa[s] > 0 and sb[s] > 0]
    return d


def scales():
    s = json.load(open(os.path.join(HERE, "scales.json")))
    return {i: s[i]["ave"] / s[i]["k"] for i in ITEMS}, {i: s[i]["phi"] for i in ITEMS}


def main():
    data, cvs, ok, failed = summary("runs_c2_h200.csv")
    print("UR-R1 runs per item (C2, H200):", sorted({len(v) for v in data.values()}), "items", len(data))
    print("UR-R2 CV of volume by item (%):", {i: round(cvs[i], 2) for i in ITEMS})
    print(f"UR-R3 CV over items: median {st.median(cvs.values()):.2f}, mean {st.mean(cvs.values()):.2f}, "
          f"min {min(cvs.values()):.2f} (item {min(cvs, key=cvs.get)}), max {max(cvs.values()):.2f} (item {max(cvs, key=cvs.get)}); "
          f"items with CV below 2%: {sum(c < 2 for c in cvs.values())}, below 5%: {sum(c < 5 for c in cvs.values())}")
    print("UR-R4 failed runs (no closed surface):", failed)
    band = [1.96 * math.sqrt(2) * cvs[i] for i in ITEMS]
    diff = [abs(a - b) for a, b in zip(APE_VOLETAPP, APE_VOLTEX)]
    print(f"UR-R5 two single runs of the same configuration differ by up to 1.96*sqrt(2)*CV (95%): median over items "
          f"{st.median(band):.2f} points; |APE VolETA++ - VolTex| median {st.median(diff):.2f} points, "
          f"within the band on {sum(d < b for d, b in zip(diff, band))} of 13 items")
    print(f"UR-R6 published APE of VolETA++ against CV: median APE {st.median(APE_VOLETAPP):.2f}; "
          f"APE below 2*CV on {sum(a < 2 * cvs[i] for a, i in zip(APE_VOLETAPP, ITEMS))} of 13 items")
    rms = math.sqrt(st.mean(c ** 2 for c in cvs.values()))
    print(f"UR-R9 run-to-run SD of the mean error over 13 items, about RMS(CV) / sqrt(13): {rms / math.sqrt(13):.2f} points "
          f"(MAPE VolTex 4.46 against VolETA++ 4.26: 0.20 points)")
    print(f"UR-R12 SD of the difference between two single-run MAPEs, about sqrt(2) times UR-R9: {math.sqrt(2) * rms / math.sqrt(13):.2f} points "
          f"(P4 mask ablation: 6.10 against 4.26, 1.84 points)")
    d = same_seed("runs_c2_3090.csv", "runs_c2_h200.csv")
    print(f"UR-R7 same item and seed, RTX 3090 against H200: {len(d)} pairs, median {st.median(d):.2f}%, max {max(d):.2f}%")
    A, B = runs("runs_c2_3090.csv"), runs("runs_c2_h200.csv")
    sg = [100 * (dict(B[i])[k] - v) / v for i in ITEMS for k, v in A.get(i, []) if k in dict(B.get(i, [])) and v > 0]
    print(f"UR-R11 signed, H200 minus RTX 3090: mean {st.mean(sg):+.2f}% (SE {st.stdev(sg) / math.sqrt(len(sg)):.2f}), "
          f"median {st.median(sg):+.2f}%, positive {sum(x > 0 for x in sg)} of {len(sg)}")
    rows = list(csv.DictReader(open(os.path.join(HERE, "runs_c2_h200.csv"))))
    opened = [r for r in rows if int(r["open_faces"]) > 0]
    print(f"UR-R10 runs with an open surface left out: {len(opened)} of {len(rows)}, on "
          f"{len({r['item'] for r in opened})} of 13 items")
    c1, cv1, ok1, f1 = summary("runs_c1_h100.csv")
    print(f"UR-R8 C1 (NeuS2 scale 0.33, H100): median CV {st.median(cv1.values()):.2f}, mean {st.mean(cv1.values()):.2f}; "
          f"failed runs {f1}; median CV without failed runs {st.median(ok1.values()):.2f}")
    s, phi = scales()
    mean_v = {i: st.mean(v for _, v in data[i]) for i in ITEMS}
    implied = {i: (GT[i] * 1e-6 / mean_v[i]) ** (1 / 3) for i in ITEMS}
    print(f"UR-S1 threshold used (240 where it finds at least 10 boards, else the one finding the most): 240 on {sum(phi[i] == 240 for i in ITEMS)} items, "
          f"220 on {sum(phi[i] == 220 for i in ITEMS)}, 210 on {sum(phi[i] == 210 for i in ITEMS)}")
    print(f"UR-S2 regenerated scale (l / med(d), NeuS2 units): mean {st.mean(s.values()):.4f}, SD {st.stdev(s.values()):.4f}; "
          f"Table 4: {TABLE4_MEAN} ({TABLE4_SD})")
    print(f"UR-S4 scale implied by the published errors' ground truth for the C2 reconstructions: mean "
          f"{st.mean(implied.values()):.4f}, SD {st.stdev(implied.values()):.4f}; regenerated over implied: "
          f"{min(s[i] / implied[i] for i in ITEMS):.3f} to {max(s[i] / implied[i] for i in ITEMS):.3f}")
    apes = [100 * abs(mean_v[i] * s[i] ** 3 * 1e6 - GT[i]) / GT[i] for i in ITEMS]
    if os.path.exists(os.path.join(HERE, "runs_c2t_mixed.csv")):
        for name, label in (("runs_c2t_mixed.csv", "C2t (VolTex settings)"), ("runs_c3_mixed.csv", "C3 (VolTex inputs and settings)")):
            if os.path.exists(os.path.join(HERE, name)):
                _, c, _, f = summary(name)
                print(f"UR-F0 {label}: median CV {st.median(c.values()):.2f}%, mean {st.mean(c.values()):.2f}%, failed runs {f}; "
                      f"by item {{{', '.join(f'{i}: {c[i]:.1f}' for i in ITEMS)}}}")
        factor("runs_c2_h200.csv", "runs_c2t_mixed.csv", "training settings, from VolTex (5,000 at 256, C2t) to VolETA++ (15,000 at 512, C2), same inputs", "UR-F1")
        if os.path.exists(os.path.join(HERE, "runs_c3_mixed.csv")):
            factor("runs_c2t_mixed.csv", "runs_c3_mixed.csv", "masks and keyframes, from DEVA (VolTex, C3) to stored (VolETA++, C2t), VolTex settings", "UR-F2")
    print(f"UR-S5 MAPE with regenerated scales on the C2 reconstructions: {st.mean(apes):.2f}% (published 4.26)")


def perm_p(x, y):
    """Two-sided exact permutation test of a difference in means (5 against 5: 252 splits)."""
    pool, obs = x + y, abs(st.mean(x) - st.mean(y))
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), len(x)):
        a = [pool[i] for i in idx]
        b = [pool[i] for i in range(len(pool)) if i not in idx]
        hits += abs(st.mean(a) - st.mean(b)) >= obs - 1e-12
        tot += 1
    return hits / tot


def factor(name_a, name_b, label, uid):
    """Relative change of the mean volume from configuration b to a, per item, with its tests."""
    A, B = runs(name_a), runs(name_b)
    rel, ps, ma, mb = [], [], [], []
    for i in ITEMS:
        x = [v for _, v in A[i] if v > 0]
        y = [v for _, v in B[i] if v > 0]
        rel.append(100 * (st.mean(x) - st.mean(y)) / st.mean(y))
        ps.append(perm_p(x, y))
        ma.append(st.mean(x)); mb.append(st.mean(y))
    # across items on log ratios: each item has its own NeuS2 units, so raw volume differences are not comparable
    n, w, p = wilcoxon_exact([math.log(a / b) for a, b in zip(ma, mb)], [0.0] * len(ma))
    _, ca, _, _ = summary(name_a); _, cb, _, _ = summary(name_b)
    na = {i: len([v for _, v in A[i] if v > 0]) for i in ITEMS}; nb = {i: len([v for _, v in B[i] if v > 0]) for i in ITEMS}
    floor = st.median(math.sqrt(2 / math.pi) * math.sqrt(ca[i] ** 2 / na[i] + cb[i] ** 2 / nb[i]) for i in ITEMS)
    print(f"{uid} {label}: change of the mean volume by item (%): {[round(r, 1) for r in rel]}; median |change| "
          f"{st.median(abs(r) for r in rel):.2f}%, median {st.median(rel):+.2f}%; items with permutation p < 0.05: "
          f"{sum(q < 0.05 for q in ps)} of 13; Wilcoxon over items on log ratios n={n}, W+={w:g}, p={p:.4f}; "
          f"median absolute change expected from run-to-run variation alone (noise floor) {floor:.2f}%; "
          f"lowered on {sum(r < 0 for r in rel)} of 13")
    return rel


def selftest():
    assert abs(cv([1.0, 1.1, 0.9]) - 10.0) < 1e-9
    assert abs(perm_p([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) - 2 / 252) < 1e-12  # only the two extreme splits
    print("selftest ok")


if __name__ == "__main__":
    selftest()
    main()
