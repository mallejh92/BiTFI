"""Print the LaTeX rows of Tables S7, S11 and S12 for the validation-tuned LightGBM control (CEA revision).

Read-only: every value comes from analysis/revision3/{covreg,indicators}/ written by
revision3_covreg_tuning.py --tables/--indicators.  Formats follow supplementary.tex.
"""
import json
import os
from pathlib import Path

import pandas as pd

RUN = Path(os.environ.get("BITFI_RUN_ROOT", Path(__file__).resolve().parents[1] / "03_result/revision2_20261019"))
C = RUN / "analysis/revision3/covreg"
I = RUN / "analysis/revision3/indicators"
REF, LG, RI, LG0, TF = "BiTFI-TimesFM3", "CovReg-LGBM-tuned", "CovReg-Ridge", "CovReg-LGBM", "TimesFM3.0-COV-SPA"
cov = pd.read_csv(RUN / "figures/source_data/Figure1_profile/coverage.csv")
order = list(dict.fromkeys(cov[cov.role == "test"].name))
TEST = {n: f"Test~{i:02d}" for i, n in enumerate(order, 1)}


def num(x, d=2):
    s = f"{x:.{d}f}"
    return f"${s}$" if s.startswith("-") else s


def ci(lo, hi, d=2):
    return f"[{num(lo, d)}, {num(hi, d)}]"


def p4(x):
    return f"{x:.4f}"


out = []
# ---------- S11 (A) ----------
summ = pd.read_csv(C / "covreg_summary.csv").set_index("model")
ctx = pd.read_csv(C / "covreg_context_summary.csv").set_index("model")
fam = pd.read_csv(C / "covreg_paired.csv").set_index("reference")
unt = pd.read_csv(C / "covreg_paired_untuned.csv").iloc[0]
sites = pd.read_csv(C / "covreg_sites.csv").set_index("greenhouse")
out.append("% S11 (A)")
for m, label in [(LG, "LightGBM (local + cross)"), (RI, "Ridge (local + cross)")]:
    r, f = summ.loc[m], fam.loc[m]
    lower = int((sites[REF] < sites[m]).sum())
    out.append(f"{label} & {r['mean']:.4f} & {ci(r.low, r.high, 4)} & {f.reduction:.2f} & {ci(f.low, f.high)} & "
               f"{p4(f.p)} & {p4(f.holm_p)} & {lower}/10 \\\\")
r = summ.loc[LG0]
out.append(f"LightGBM (local + cross), a priori settings$^{{\\ddagger}}$ & {r['mean']:.4f} & {ci(r.low, r.high, 4)} & "
           f"{unt.reduction:.2f} & {ci(unt.low, unt.high)} & {p4(unt.p)} & --- & {int((sites[REF] < sites[LG0]).sum())}/10 \\\\")
pc = pd.read_csv(C / "covreg_paired_context.csv")
tf = pc[(pc.model == LG) & (pc.reference == TF)].iloc[0]
lr = pd.read_csv(C / "covreg_paired_lgbm_vs_ridge.csv").set_index("reference")
out.append(f"% footnote: tuned vs TimesFM3 {tf.reduction:.2f} [{tf.low:.2f}, {tf.high:.2f}] p={tf.p:.4f}; "
           f"tuned vs ridge {lr.loc[RI].reduction:.2f} [{lr.loc[RI].low:.2f}, {lr.loc[RI].high:.2f}] p={lr.loc[RI].p:.4f}; "
           f"tuned vs a priori {lr.loc[LG0].reduction:.2f} [{lr.loc[LG0].low:.2f}, {lr.loc[LG0].high:.2f}] p={lr.loc[LG0].p:.4f}")
higher = [f"{TEST[g]} ({sites.loc[g, REF]:.4f} versus {sites.loc[g, LG]:.4f})" for g in sites.index if sites.loc[g, REF] > sites.loc[g, LG]]
out.append("% BiTFI higher than tuned at: " + "; ".join(sorted(higher)))
out.append(f"% tuned lower than TimesFM3 at {(sites[LG] < sites[TF] if TF in sites else pd.Series(dtype=bool)).sum() if TF in sites else 'n/a'} sites")

# ---------- S11 (B) ----------
sg = pd.read_csv(C / "covreg_scenario_gap.csv")
sc = pd.read_csv(C / "covreg_scenario.csv")
gp = pd.read_csv(C / "covreg_gap.csv")
st = pd.read_csv(C / "covreg_paired_strata.csv")
st = st[(st.model == REF) & (st.reference == LG)]
ctxsg = None
tfsg = pd.read_csv(RUN / "analysis/all_gaps.csv")
tfsc = pd.read_csv(RUN / "analysis/all_scenarios.csv")


def val(frame, model, **k):
    q = frame[frame.model == model]
    for a, b in k.items():
        q = q[q[a] == b]
    return float(q["mean"].iloc[0])


out.append("% S11 (B)")
for s in "ABC":
    first = True
    for g in [6, 12, 24, 72, 168, "All"]:
        if g == "All":
            vals = [val(sc, REF, scenario=s), val(sc, LG, scenario=s), val(sc, RI, scenario=s), val(tfsc, TF, scenario=s)]
            q = st[(st.stratum == "scenario") & (st.scenario == s)].iloc[0]
        else:
            vals = [val(sg, REF, scenario=s, gap_length_h=g), val(sg, LG, scenario=s, gap_length_h=g),
                    val(sg, RI, scenario=s, gap_length_h=g), val(tfsg, TF, scenario=s, gap_length_h=g)]
            q = st[(st.stratum == "scenarioxgap_length_h") & (st.scenario == s) & (st.gap_length_h == g)].iloc[0]
        lead = f"{s} & {g}" if first else f" & {g}"
        first = False
        out.append(lead + " & " + " & ".join(f"{v:.4f}" for v in vals) +
                   f" & {num(q.reduction)} & {ci(q.low, q.high)} & {p4(q.p)} \\\\")
    out.append("\\addlinespace")
from analyze_revision_experiments import summary as _summary  # noqa: E402
_tf = pd.read_csv(RUN / "evaluation" / TF / "results.csv").query("group_type=='all'")
tfgap = _summary(_tf, extra=["gap_length_h"])
first = True
for g in [6, 12, 24, 72, 168]:
    vals = [val(gp, REF, gap_length_h=g), val(gp, LG, gap_length_h=g), val(gp, RI, gap_length_h=g),
            val(tfgap, TF, gap_length_h=g)]
    q = st[(st.stratum == "gap_length_h") & (st.gap_length_h == g)].iloc[0]
    out.append(("All & " if first else " & ") + f"{g} & " + " & ".join(f"{v:.4f}" for v in vals)
               + f" & {num(q.reduction)} & {ci(q.low, q.high)} & {p4(q.p)} \\\\")
    first = False
# ridge minimum reduction and the LightGBM pass trajectory in C
stall = pd.read_csv(C / "covreg_paired_strata.csv")
rid = stall[(stall.model == REF) & (stall.reference == RI) & (stall.stratum == "scenarioxgap_length_h")]
out.append(f"% ridge min reduction over 15 combos {rid.reduction.min():.2f}")
rs = pd.read_csv(C / "covreg_refinement_scenario.csv")
c3 = rs[(rs.model == f"{LG}-refine3") & (rs.scenario == "C")]["mean"].iloc[0]
c5 = rs[(rs.model == f"{LG}-refine5") & (rs.scenario == "C")]["mean"].iloc[0]
out.append(f"% tuned C NMAE pass3 {c3:.4f} -> pass5 {c5:.4f}")
n_pos = (st[st.stratum == "scenarioxgap_length_h"].reduction > 0).sum()
n_zero = (st[st.stratum == "scenarioxgap_length_h"].low <= 0).sum()
out.append(f"% BiTFI lower mean in {n_pos}/15 combos; CI includes zero in {n_zero}/15")

# ---------- S11 (C) ----------
mae = pd.read_csv(C / "covreg_variables_MAE.csv")
tfm = pd.read_csv(RUN / "analysis/all_physical_MAE.csv")
VARS = [("Tin", "$T_{\\mathrm{in}}$ ($^\\circ$C)"), ("Tout", "$T_{\\mathrm{out}}$ ($^\\circ$C)"), ("RH", "RH (pp)"),
        ("CO2", "CO$_2$ (ppm)"), ("Rad", "Rad (W\\,m$^{-2}$)")]
out.append("% S11 (C)")
for v, label in VARS:
    vals = [val(mae, REF, variable=v), val(mae, LG, variable=v), val(mae, RI, variable=v), val(tfm, TF, variable=v)]
    q = st[(st.stratum == "variable") & (st.variable == v)].iloc[0]
    out.append(f"{label} & " + " & ".join(f"{x:.2f}" for x in vals) + f" & {num(q.reduction)} & {ci(q.low, q.high)} & {p4(q.p)} \\\\")
rv = stall[(stall.model == REF) & (stall.reference == RI) & (stall.stratum == "variable")]
out.append(f"% ridge variable reductions {rv.reduction.min():.2f} ({rv.loc[rv.reduction.idxmin(), 'variable']}) to "
           f"{rv.reduction.max():.2f} ({rv.loc[rv.reduction.idxmax(), 'variable']})")

# ---------- S7 family (f) ----------
loo = pd.read_csv(I / "loo_effect_sizes_covreg.csv").set_index("reference")
out.append("% S7 family (f)")
for m, label in [(LG, "LightGBM (local + cross)"), (RI, "Ridge (local + cross)")]:
    r = loo.loc[m]
    out.append(f"{label} & {r.reference_mean:.4f} & {r.reduction:.2f} & {ci(r.low, r.high)} & {p4(r.holm_p)} & "
               f"{r.loo_min:.2f} ({TEST[r.loo_min_excluded]}) & {r.loo_max:.2f} ({TEST[r.loo_max_excluded]}) & "
               f"{num(r.site_reduction_min)} & {num(r.site_reduction_max)} & {int(r.sites_model_lower)}/10 \\\\"
               + ("" if r.model_lower_all_loo else " % NOT lower in all LOO"))

# ---------- S12 covreg rows ----------
out.append("% S12")
for k, metrics, scopes in [("vpd", ["VPD_MAE", "VPD_daily_AE"], ["B", "C", "B+C"]), ("radiation", ["H_AE"], ["A", "C", "A+C"])]:
    p = pd.read_csv(I / f"indicators_{k}_paired_covreg.csv")
    p = p[p.variant == "primary"]
    for met in metrics:
        out.append(f"% {k} {met}")
        d = 4 if k == "vpd" else 3
        for m, label in [(LG, "LightGBM (local + cross)"), (RI, "Ridge (local + cross)")]:
            cells = []
            for sc_ in scopes:
                r = p[(p.metric == met) & (p.scope == sc_) & (p.reference == m)].iloc[0]
                if "+" in sc_:
                    cells.append(f"{r.reference_mean:.{d}f} & {num(r.reduction)} & {p4(r.holm_p)}")
                else:
                    cells.append(f"{r.reference_mean:.{d}f} & {num(r.reduction)} & {ci(r.low, r.high)} & {p4(r.holm_p)}")
            out.append(f"{label} & " + " & ".join(cells) + " \\\\")
print("\n".join(out))
