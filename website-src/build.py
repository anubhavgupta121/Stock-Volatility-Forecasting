"""Build the chart-led version of the site.
Splices src3/main.html into src3/page.html, inlines the forecasts for the interactive charts,
and writes every key table and number into the HTML itself, so the page reads fully without JavaScript.
Inputs: data/test_predictions.csv and data/val_predictions.csv, the forecasts exported by notebooks/checks.ipynb
Output: index.html (on the gh-pages branch: python website-src/build.py).
Every number on the page is either computed here from those two files or copied from the notebook's printed output."""
import csv, json, math, pathlib, re
root = pathlib.Path(__file__).resolve().parent.parent
src = pathlib.Path(__file__).resolve().parent
DEV = (root / "site").is_dir()                     # working copy layout: site/data, site/index.html
DATA = root / ("site/data" if DEV else "data")


def load(name):
    rows = list(csv.reader(open(DATA / name)))
    cols = ["date"] + [{"HAR + range + VIX": "HAR + VIX"}.get(c, c) for c in rows[0][1:]]   # either CSV header works
    return {"cols": cols, "rows": [[r[0]] + [float(f"{float(v):.6g}") for v in r[1:]] for r in rows[1:]]}


def records(d):
    return [dict(zip(d["cols"], r)) for r in d["rows"]]


test, val = load("test_predictions.csv"), load("val_predictions.csv")
T, V = records(test), records(val)
for r in V:
    r["nov"], r["vix"] = r["XGBoost-QLIKE (no VIX)"], r["XGBoost-QLIKE (with VIX)"]

MODELS = [("XGBoost-QLIKE", "XGBoost (QLIKE loss)", "--s1"), ("GARCH", "GARCH(1,1)", "--s2"), ("HAR + VIX", "HAR + range + VIX", "--s3"),
          ("Random forest", "Random forest", "--s4"), ("XGBoost", "XGBoost (squared error)", "--s5"), ("HAR", "HAR", "--s6"),
          ("EWMA", "EWMA", "--s7"), ("Rolling (22d)", "Rolling (22-day)", "--s8"), ("Naive (5d)", "Naive (5-day)", "--s9")]
LAB = {k: l for k, l, _ in MODELS}
COL = {k: c for k, _, c in MODELS}


def loss(r, k):
    x = (r["actual"] / r[k]) ** 2
    return x - math.log(x) - 1


def qlike(rs, k): return sum(loss(r, k) for r in rs) / len(rs)
def rmse(rs, k): return math.sqrt(sum((r[k] - r["actual"]) ** 2 for r in rs) / len(rs))
def f3(v): return f"{v:.3f}"
def pct(v): return f"{v * 100:.2f}%"
def sw(c, op=""): return f'<span class="sw" style="background:var({c}){op}"></span>'
def year(r): return int(r["date"][:4])


def heat(v, ref, bold=False):
    """Cell attributes: blue when better than the reference (GARCH), red when worse; stronger = bigger gap."""
    r = v / ref
    cls = ' class="b"' if bold else ""
    if abs(r - 1) < 0.004: return cls
    a = min(1, (1 - r) / 0.45) * 52 if r < 1 else min(1, (r - 1) / 1.0) * 52
    return f'{cls} style="background:color-mix(in srgb, var({"--good" if r < 1 else "--bad"}) {max(a, 9):.0f}%, transparent)"'


def quantile(xs, p):  # same rule as d3.quantile
    xs = sorted(xs); h = (len(xs) - 1) * p; lo = math.floor(h)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (h - lo)


def fmt_date(s):
    y, m, d = s.split("-"); M = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"{int(d)} {M[int(m) - 1]} {y}"


out = {}
years = sorted({year(r) for r in T})
res = sorted([(k, qlike(T, k), rmse(T, k), [qlike([r for r in T if year(r) == y], k) for y in years]) for k, _, _ in MODELS], key=lambda t: t[1])
g = next(t[1] for t in res if t[0] == "GARCH")
best = [min(t[3][i] for t in res) for i in range(len(years))]
gy = next(t[3] for t in res if t[0] == "GARCH")

# full test table (§6.2)
h = "<thead><tr><th>Model</th><th>QLIKE</th><th>×&nbsp;GARCH</th><th>RMSE</th>" + "".join(f"<th>{y}{'*' if y == 2026 else ''}</th>" for y in years) + "</tr></thead><tbody>"
for i, (k, q, e, ys) in enumerate(res):
    h += f'<tr{' class="best"' if i == 0 else ""}><td>{sw(COL[k])}{LAB[k]}</td><td{heat(q, g)}>{f3(q)}</td><td>{q / g:.2f}</td><td>{e:.4f}</td>' + "".join(f'<td{heat(v, gy[j], v == best[j])}>{f3(v)}</td>' for j, v in enumerate(ys)) + "</tr>"
out["test_table"] = h + "</tbody>"

# block-bootstrap confidence intervals for the main test differences (same method as the notebook's check cell):
# daily QLIKE differences, 22-day blocks because neighbouring targets overlap, 5000 resamples, seed 42
import numpy as np
yt = np.array([r["actual"] for r in T])
def daily(k):
    x = (yt / np.array([r[k] for r in T])) ** 2
    return x - np.log(x) - 1
nT, BL, RS = len(T), 22, 5000
rng = np.random.default_rng(42)
starts = rng.integers(0, nT - BL + 1, size=(RS, -(-nT // BL)))
idx = (starts[:, :, None] + np.arange(BL)).reshape(RS, -1)[:, :nT]
base = daily("XGBoost-QLIKE")
CI = {}
for k in ["GARCH", "Random forest", "XGBoost", "HAR + VIX"]:
    d = daily(k) - base
    bt = d[idx].mean(axis=1)
    CI[k] = (float(d.mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5)))
h = '<thead><tr><th>XGBoost-QLIKE against</th><th>QLIKE gap</th><th>95% interval</th><th class="l">Reading</th></tr></thead><tbody>'
def fm(v): return f3(v).replace("-", "−")
def reading(lo):
    if lo >= 0.0005: return "<b>Real</b>: the whole interval is above 0"
    if lo > -0.0005: return "Borderline: the interval just reaches 0"
    return "Could be luck: the interval includes 0"
for k, (m, lo, hi) in CI.items():
    h += (f'<tr><td>{sw(COL[k])}{LAB[k]}</td><td>{fm(m)}</td><td style="white-space:nowrap">{fm(lo)} to {fm(hi)}</td>'
          f'<td class="l">{reading(lo)}</td></tr>')
print({k: [round(x, 5) for x in v] for k, v in CI.items()})
out["ci_table"] = h + "</tbody>"
out["ci_note"] = (f"The gap to GARCH is {f3(CI['GARCH'][0])}, and even the low end of its interval ({f3(CI['GARCH'][1])}) is well above zero, so that win is real. "
                  f"The gap to the forest ({fm(CI['Random forest'][0])}, interval {fm(CI['Random forest'][1])} to {fm(CI['Random forest'][2])}) "
                  f"{'stays above zero' if CI['Random forest'][1] > 0 else 'includes zero'}, "
                  f"{'so even that win holds up' if CI['Random forest'][1] > 0 else 'so XGBoost and the forest cannot be told apart on these years'}.")

# abstract table
y25 = [r for r in T if year(r) == 2025]
h = '<table class="bk"><thead><tr><th>Model, test 2023–2026</th><th>QLIKE</th><th>×&nbsp;GARCH</th><th>RMSE</th><th>2025</th></tr></thead><tbody>'
for k in ["XGBoost-QLIKE", "Random forest", "HAR + VIX", "GARCH", "Naive (5d)"]:
    q = qlike(T, k)
    h += f'<tr{' class="best"' if k == "XGBoost-QLIKE" else ""}><td>{LAB[k]}</td><td>{f3(q)}</td><td>{q / g:.2f}</td><td>{rmse(T, k):.4f}</td><td>{f3(qlike(y25, k))}</td></tr>'
out["abstract_table"] = h + "</tbody></table>"

# explorer range table, initial state (all dates; three models shown)
shown = {"XGBoost-QLIKE", "GARCH", "HAR + VIX"}
h = "<thead><tr><th>Model, selected dates</th><th>QLIKE</th><th>×&nbsp;GARCH</th><th>RMSE</th></tr></thead><tbody>"
for i, (k, q, e, _) in enumerate(res):
    on = k in shown
    h += f'<tr class="{"best" if i == 0 else ""}{"" if on else " off"}"><td>{sw(COL[k])}{LAB[k]}{" · best" if i == 0 else ""}</td><td>{f3(q)}</td><td>{q / g:.2f}</td><td>{e:.4f}</td></tr>'
out["range_table"] = h + "</tbody>"

# validation bars (from the notebook's validation runs)
VB = [("XGBoost (QLIKE loss)", "HAR inputs + range + VIX · chosen", 0.400, True), ("Random forest", "HAR inputs + range + VIX", 0.416, False),
      ("XGBoost (squared error)", "HAR inputs + range + VIX", 0.416, False), ("HAR + range + VIX", "Linear, same inputs", 0.42, False),
      ("XGBoost (QLIKE loss)", "HAR inputs + range", 0.458, False), ("GARCH(1,1)", "Baseline", 0.494, False),
      ("XGBoost (QLIKE loss)", "HAR inputs only", 0.497, False), ("XGBoost (squared error)", "HAR inputs only", 0.516, False),
      ("HAR", "HAR inputs only", 0.545, False), ("Random forest", "HAR inputs only", 0.547, False), ("EWMA", "Baseline", 0.556, False),
      ("Rolling (22-day)", "Baseline", 0.631, False)]
mx, ref = 0.68, 0.494
h = ""
for l, s, v, hi in VB:
    w = v / mx * 82
    h += (f'<div class="brow"><div class="lab"><b>{l}</b><span>{s}</span></div><div class="track"><div class="refline" style="left:{ref / mx * 82:.2f}%"></div>'
          f'<div class="fill{" hi" if hi else ""}" style="width:{w:.2f}%"></div><span class="val" style="left:calc({w:.2f}% + 6px)">{v:.3f}</span></div></div>')
out["val_bars"] = h + '<p class="note">Naive (5-day) scored 1.008 and is left off the scale.</p>'

# before/after VIX: crash window (initial state of the interactive figure)
NAME = {"GARCH": "GARCH(1,1)", "nov": "XGBoost-QLIKE without VIX", "vix": "XGBoost-QLIKE with VIX"}
W = [r for r in V if "2020-02-03" <= r["date"] <= "2020-05-29"]
q = {k: qlike(W, k) for k in ["GARCH", "nov", "vix"]}
mxs = {k: max(r[k] for r in W) for k in ["GARCH", "nov", "vix", "actual"]}
out["vix_note"] = (f"Over these four months VIX made no difference: {f3(q['nov'])} without it, {f3(q['vix'])} with it, against {f3(q['GARCH'])} for GARCH. "
                   f"Both versions hit a flat top on 12 March. Switch versions and watch the dashed line: VIX lifted the top only from {pct(mxs['nov'])} to {pct(mxs['vix'])}, "
                   f"while actual volatility peaked at {pct(mxs['actual'])}.")
bq = min(q.values())
h = "<thead><tr><th>Model, Feb–May 2020</th><th>QLIKE</th><th>×&nbsp;GARCH</th><th>Peak forecast</th></tr></thead><tbody>"
for k in ["GARCH", "nov", "vix"]:
    dim = k == "vix"
    h += f'<tr class="{"best" if q[k] == bq else ""}{" off" if dim else ""}"><td>{sw("--s2" if k == "GARCH" else "--s1", ";opacity:.45" if dim else "")}{NAME[k]}{" · shown" if k == "nov" else ""}</td><td>{f3(q[k])}</td><td>{q[k] / q["GARCH"]:.2f}</td><td>{pct(mxs[k])}</td></tr>'
out["vix_table"] = h + f'<tr class="off"><td>Actual</td><td></td><td></td><td>{pct(mxs["actual"])}</td></tr></tbody>'

vy = [2019, 2020, 2021, 2022]
by = {k: [qlike([r for r in V if year(r) == y], k) for y in vy] + [qlike(V, k)] for k in ["GARCH", "nov", "vix"]}
cb = [min(by[k][j] for k in by) for j in range(5)]
h = "<thead><tr><th>Validation QLIKE</th>" + "".join(f"<th>{y}</th>" for y in vy) + "<th>All</th></tr></thead><tbody>"
for k in ["GARCH", "nov", "vix"]:
    h += f'<tr><td>{sw("--s2" if k == "GARCH" else "--s1", ";opacity:.45" if k == "nov" else "")}{NAME[k]}</td>' + "".join(f'<td{heat(v, by["GARCH"][j], v == cb[j])}>{f3(v)}</td>' for j, v in enumerate(by[k])) + "</tr>"
out["vix_years"] = h + "</tbody>"

BANDS = [("below 0.7%", 0, 0.007), ("0.7–1.2%", 0.007, 0.012), ("1.2–2%", 0.012, 0.02), ("2–3%", 0.02, 0.03), ("above 3%", 0.03, 1)]
bands = []
for l, a, b in BANDS:
    rs = [r for r in V if a <= r["actual"] < b]
    bands.append((l, len(rs), qlike(rs, "nov"), qlike(rs, "vix"), sum(loss(r, "nov") - loss(r, "vix") for r in rs)))
tot = sum(b[4] for b in bands); smax = max(b[4] / tot for b in bands)
h = "<thead><tr><th>Actual volatility</th><th>Days</th><th>Without VIX</th><th>With VIX</th><th>Share of VIX's gain</th></tr></thead><tbody>"
for l, n, a, b, gn in bands:
    sh = gn / tot
    h += f'<tr{' class="best"' if sh == smax else ""}><td>{l}</td><td>{n}</td><td>{f3(a)}</td><td>{f3(b)}</td><td><span class="share"><span class="shb" style="width:{max(0, sh / smax * 56):.1f}px"></span>{round(sh * 100)}%</span></td></tr>'
out["vix_bands"] = h + "</tbody>"

# hybrid table (§4.7), numbers from the notebook's validation runs
HY = [("GARCH(1,1)", [0.494, 0.488, 0.730, 0.510, 0.244]), ("Hybrid, HAR inputs + range", [0.467, 0.442, 0.724, 0.474, 0.229]),
      ("XGBoost-QLIKE, HAR inputs + range", [0.458, 0.446, 0.695, 0.454, 0.235]), ("Hybrid, + VIX", [0.435, 0.399, 0.686, 0.435, 0.216]),
      ("XGBoost-QLIKE, + VIX", [0.400, 0.351, 0.630, 0.395, 0.222])]
h = "<thead><tr><th>Validation QLIKE</th><th>All</th><th>2019</th><th>2020</th><th>2021</th><th>2022</th></tr></thead><tbody>"
for n, vs in HY:
    h += f'<tr{' class="best"' if n.endswith("+ VIX") and n.startswith("XGB") else ""}><td>{n}</td>' + "".join(f"<td{heat(v, HY[0][1][j])}>{v:.3f}</td>" for j, v in enumerate(vs)) + "</tr>"
out["hybrid_table"] = h + "</tbody>"

# volatility clustering (§1): this week against next week, 2019-2026
seq = sorted([(r["date"], r["actual"]) for r in V] + [(r["date"], r["actual"]) for r in T])
av = [v for _, v in seq]
pairs = [(av[i], av[i + 5]) for i in range(0, len(av) - 5, 5)]
lx, ly = [math.log(a) for a, _ in pairs], [math.log(b) for _, b in pairs]
mx_, my_ = sum(lx) / len(lx), sum(ly) / len(ly)
corr = sum((a - mx_) * (b - my_) for a, b in zip(lx, ly)) / math.sqrt(sum((a - mx_) ** 2 for a in lx) * sum((b - my_) ** 2 for b in ly))
lo3, hi3 = quantile(av, 1 / 3), quantile(av, 2 / 3)
top = [q for q in pairs if q[0] > hi3]; bot = [q for q in pairs if q[0] < lo3]
out["cluster_note"] = (f"Each dot is one week of 2019–2026 ({len(pairs)} weeks): its volatility across, the next week's up, both on log scales. "
                       f"After a week in the top third, the next week was also in the top third {round(sum(1 for q in top if q[1] > hi3) / len(top) * 100)}% of the time; "
                       f"after a bottom-third week, the next stayed in the bottom third {round(sum(1 for q in bot if q[1] < lo3) / len(bot) * 100)}% of the time "
                       f"(a coin-flip world would give 33%). The correlation of the logs is {corr:.2f}. This memory is what every model here learns from.")

# errors (§7)
K = "XGBoost-QLIKE"
L = [(i, r, loss(r, K)) for i, r in enumerate(T)]
total = sum(l for _, _, l in L)
srt = sorted(L, key=lambda t: -t[2])
top = srt[:round(len(T) * 0.05)]
under = [t for t in L if t[1]["actual"] > t[1][K]]
out["err_summary"] = (f"For the winning model, the worst 5% of test days ({len(top)} of {len(T)}) account for <strong>{round(sum(t[2] for t in top) / total * 100)}%</strong> of its total QLIKE. "
                      f"It under-predicted on {round(len(under) / len(T) * 100)}% of days, and those days carry {round(sum(t[2] for t in under) / total * 100)}% of the loss: "
                      f"the misses that hurt are the sudden jumps it did not see coming.")
worst = sorted(srt[:40], key=lambda t: t[0])
eps = []
for t in worst:
    if eps and t[0] - eps[-1]["end"] <= 3:
        eps[-1]["end"] = t[0]; eps[-1]["days"].append(t)
    else:
        eps.append({"start": t[0], "end": t[0], "days": [t]})
for e in eps:
    e["loss"] = sum(t[2] for t in e["days"]); e["peak"] = max(e["days"], key=lambda t: t[2])
eps.sort(key=lambda e: -e["loss"])
NOTES = [("2025-03-20", "2025-04-10", "The 2 April 2025 tariff announcement. The target already contains the crash week, so these days look like misses before the news existed."),
         ("2024-07-22", "2024-08-02", "The week before the 5 August 2024 global sell-off, when VIX briefly spiked above 60."),
         ("2024-12-05", "2024-12-20", "The run-up to the 18 December 2024 Fed meeting, after which the S&amp;P fell about 3% in a day.")]
h = ""
for e in eps[:5]:
    a, b, p = T[e["start"]]["date"], T[e["end"]]["date"], e["peak"][1]
    note = next((n for lo, hi, n in NOTES if lo <= p["date"] <= hi), "")
    lo, hi = T[max(0, e["start"] - 20)]["date"], T[min(len(T) - 1, e["end"] + 20)]["date"]
    nd = len(e["days"])
    h += (f'<button type="button" class="ep" data-lo="{lo}" data-hi="{hi}"><span class="t">{fmt_date(a)}{" – " + fmt_date(b) if e["end"] > e["start"] else ""}</span>'
          f'<span class="m">{nd} day{"s" if nd > 1 else ""} · loss {e["loss"]:.1f}</span>'
          f'<span class="n">Worst day {fmt_date(p["date"])}: actual {pct(p["actual"])}, XGBoost {pct(p[K])}, GARCH {pct(p["GARCH"])}. {note}</span></button>')
out["episodes"] = h

acts = [r["actual"] for r in T]
t1, t2 = quantile(acts, 1 / 3), quantile(acts, 2 / 3)
reg = [("Calm", f"below {pct(t1)}", [r for r in T if r["actual"] < t1]), ("Normal", f"{pct(t1)}–{pct(t2)}", [r for r in T if t1 <= r["actual"] < t2]),
       ("Volatile", f"above {pct(t2)}", [r for r in T if r["actual"] >= t2])]
RM = ["XGBoost-QLIKE", "GARCH", "HAR + VIX", "Random forest"]
vals = [[qlike(rs, k) for k in RM] for _, _, rs in reg]
vmax = max(max(v) for v in vals) * 1.25
out["regime_note"] = f"Test days split into thirds by actual next-week volatility ({len(reg[0][2])} days each), with QLIKE for the four strongest models in each third."
h = ""
for gi, (n, d, _) in enumerate(reg):
    h += f'<div class="rgrp"><div class="rlab">{n} · {d}</div>'
    for mi, k in enumerate(RM):
        v = vals[gi][mi]; bst = v == min(vals[gi])
        h += (f'<div class="brow"><div class="lab"><span>{sw(COL[k])}{LAB[k]}</span></div><div class="track"><div class="fill" style="width:{v / vmax * 82:.1f}%;background:var({COL[k]});opacity:1"></div>'
              f'<span class="val{" bv" if bst else ""}" style="left:calc({v / vmax * 82:.1f}% + 6px)">{v:.3f}</span></div></div>')
    h += "</div>"
out["regime_bars"] = h


def median(xs):
    xs = sorted(xs); n = len(xs); return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


c, _, v = vals
gain = [sum(loss(r, "GARCH") - loss(r, K) for r in rs) for _, _, rs in reg]
med = median([r[K] / r["actual"] for r in T])
mc = lambda k: median([r[k] / r["actual"] for r in reg[0][2]])
out["regime_read"] = (f"Against GARCH, most of the lead ({round(gain[0] / sum(gain) * 100)}% of it) comes in calm weeks: {f3(c[0])} vs {f3(c[1])}. "
                      f"Every model forecasts too high in calm weeks, GARCH most of all, with a median forecast of {mc('GARCH'):.2f}× the actual against {mc(K):.2f}× for XGBoost. "
                      f"Against the other ML models the pattern flips. XGBoost wins volatile weeks ({f3(v[0])} vs {f3(v[2])} for HAR + range + VIX and {f3(v[3])} for the forest) "
                      f"and loses calm ones ({f3(c[0])} vs {f3(c[2])} and {f3(c[3])}), because the QLIKE loss leans towards forecasting high: over all test days its median forecast is {med:.2f}× the actual. "
                      f"That is the trade QLIKE asks for, since under-predicting risk costs more than over-predicting it.")

# assemble
page = (src / "page.html").read_text()
main = (src / "main.html").read_text().strip()
for k, v in out.items():
    tag = f"<!--@{k}-->"
    assert main.count(tag) == 1, tag
    main = main.replace(tag, v)
left = re.findall(r"<!--@\w+-->", main)
assert not left, left
page = page.replace("<!--@MAIN-->", main)
page = page.replace("/*__DATA__*/null", json.dumps(test, separators=(",", ":")))
page = page.replace("/*__VAL__*/null", json.dumps(val, separators=(",", ":")))
if DEV:
    (root / "build").mkdir(exist_ok=True)
    (root / "build/artifact-charts.html").write_text(page)
cut = page.index('<div class="bar">')
head, body = page[:cut], page[cut:]
reset = "<style>html{color-scheme:light}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + reset + head + '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
(root / ("site/index.html" if DEV else "index.html")).write_text(full)
print("ok", len(full), "bytes;", len(T), "test rows;", len(V), "validation rows")
