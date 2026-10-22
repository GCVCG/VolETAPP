"""Metric volumes of a configuration and their match with the published VolETA++ per-item errors.

Scale per item: ave = l_square / med(d), metres per COLMAP unit of the dense model (find_scale_headless.py, phi = 240);
k = NeuS2 units per COLMAP unit (colmap_to_nerf.py). Two readings of the scale step are compared:
  paper: s = ave / k             (supplementary of P4: "convert the unitless NeRF reconstruction into metres")
  code:  s = ave * (1 + 0.15 + 0.5)   (released find_scale.py, applied to NeuS2 units)
Volume in cm3 = raw NeuS2 volume * s**3 * 1e6, as the challenge evaluation (trimesh volume * 1e6).

usage: python evaluate.py results_c2.csv scales_c2.log k_c2.log
"""
import ast, csv, json, re, statistics as st, sys

GT = {"1": 38.53, "2": 280.36, "3": 249.65, "4": 295.13, "5": 392.58, "6": 218.31, "7": 368.77, "8": 173.13,
      "9": 232.74, "10": 163.23, "11": 85.18, "13": 308.28, "14": 589.82}  # P3 Table 1 (facts-p3-voltex.md)
PUBLISHED_APE = {"1": 3.971, "2": 0.671, "3": 1.607, "4": 8.611, "5": 6.359, "6": 2.704, "7": 7.814, "8": 4.418,
                 "9": 1.337, "10": 4.201, "11": 2.137, "13": 1.933, "14": 8.267}  # P4 supplementary (stats_lineage.py)


def load(results, scales, ks):
    raw = {}
    for r in csv.DictReader(open(results)):
        raw.setdefault(r["item"], []).append(float(r["volume_raw"]))
    ave = {}
    for line in open(scales):
        m = re.search(r"RESULT (\{.*\})", line)
        if m:
            d = json.loads(m.group(1))
            ave[d["item"]] = d["ave_radio"]
    k = {l.split()[0]: ast.literal_eval(l.split(" ", 1)[1])[0] for l in open(ks) if l.strip()}
    return raw, ave, k


def table(raw, ave, k):
    rows = []
    for i in sorted(raw, key=int):
        if i not in ave or i not in k:
            continue
        v = st.mean(raw[i])
        s = {"paper": ave[i] / k[i], "code": ave[i] * 1.65}
        ape = {f: 100 * abs(v * s[f] ** 3 * 1e6 - GT[i]) / GT[i] for f in s}
        implied = (GT[i] * 1e-6 / v) ** (1 / 3)
        rows.append((i, v, s["paper"], s["code"], implied, ape["paper"], ape["code"], PUBLISHED_APE[i]))
    return rows


def main(results, scales, ks):
    rows = table(*load(results, scales, ks))
    print("item  raw_vol  s_paper  s_code  s_implied  APE_paper  APE_code  APE_published")
    for r in rows:
        print("%4s %8.4f %8.4f %7.4f %9.4f %9.2f %9.2f %10.2f" % r)
    for name, col in (("paper", 5), ("code", 6)):
        mape = st.mean(r[col] for r in rows)
        gap = st.mean(abs(r[col] - r[7]) for r in rows)
        print(f"{name}: MAPE {mape:.2f}% over {len(rows)} items (published 4.26); mean |APE - published| {gap:.2f} points")
    print(f"scale factors, mean (sd): paper {st.mean(r[2] for r in rows):.4f} ({st.stdev(r[2] for r in rows):.4f}), "
          f"code {st.mean(r[3] for r in rows):.4f} ({st.stdev(r[3] for r in rows):.4f}), "
          f"implied {st.mean(r[4] for r in rows):.4f} ({st.stdev(r[4] for r in rows):.4f}); P4 Table 4: 0.1055 (0.0137)")


def selftest():
    # a raw volume of 1e-4 with s = 0.1 gives 0.1 cm3 ... choose numbers that give the ground truth exactly
    s = 0.1
    v = GT["1"] * 1e-6 / s ** 3
    r = table({"1": [v]}, {"1": s * 2}, {"1": 2.0})[0]
    assert abs(r[2] - s) < 1e-12 and r[5] < 1e-9 and abs(r[4] - s) < 1e-12, r
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--selftest"] else main(*sys.argv[1:])
