"""Revision-2 figures (2026-10-19), CPU only, no model inference.

Subcommands
-----------
existing  Re-render the result figures that contain configurations rescored by the
          M4 no-history correction (Figs. 3-5, S1, S4, S5; S6 with --with-s6) from
          the run root's re-aggregated ``analysis/`` tables. The drawing code is the
          published code in ``build_prism_figures.py``; the S4/S5 blocks are copied
          from ``build_prism_figures.supplementary`` so that the Fig. S5B title can
          follow its content. Figure 6 (author PDF),
          S2, S3 and S7-S10 are not touched. ``--aligned-labels`` (F3, 2026-10-20)
          places the panel letters as ``panel_label`` does in the new figures, on the
          title baseline with the right edge 8 pt left of the axes, instead of at the
          axes-fraction position of ``ps.panel``; nothing else in the figures changes.
new       Figures for the revision-2 controls (see ``NEW_FIGURES``). A figure is
          skipped, with a message, until all of its input files exist. The redesign
          previews of ``REDESIGN_FIGURES`` (fig6r, fig7r, s15r, s11h) run only when named
          with ``--only`` and are written to ``<out>/revision2_redesign``.
final     Manuscript versions of the adopted redesign (F3, 2026-10-20; see
          ``FINAL_FIGURES``): Fig. 6 information sources, Fig. 7 agricultural
          indicators, Fig. 8 (byte copy of the author's reconstruction-example PDF, the
          former Fig. 6) and Figs. S11 (heatmap), S12 and S15, with the variant names of
          the manuscript tables. Written under the manuscript file names to ``<out>``
          (main) and ``<out>/Supplementary``, with source data in
          ``<out>/source_data/<figure name>_*`` and the render log in
          ``<out>/revision2/render_log_final.json``. The draft outputs in
          ``<out>/revision2`` (old Figs. 7 and 8, dot-plot S11) are left as they are.
compare   Tabulate what changed between two rendered folders (source tables and
          recorded axis limits).

Inputs are read from one run root (``--run-root``, default ``$BITFI_RUN_ROOT`` or
``03_result/revision2_20261019``). Every output goes below ``--out`` (default
``<run root>/figures``). Writing into ``04_figure``, ``05_thesis`` or the read-only
``reevaluation_fullgap_20260929`` root is refused, so the script never replaces the
manuscript copies; those are updated separately after approval.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RUN = ROOT / "03_result/revision2_20261019"
PROTECTED = [ROOT / "04_figure", ROOT / "05_thesis", ROOT / "03_result/reevaluation_fullgap_20260929"]
S5B_TITLE_PUBLISHED = "Cross-greenhouse regression"
S5B_TITLE_REVISED = "Indoor and outdoor variables"
# The published matplotlib figures embed Liberation Sans (Arial was not installed when
# they were rendered). Arial installed later in ~/.fonts has no U+2082 glyph, so the
# default style would print "CO  concentration"; pin the published family instead.
PUBLISHED_FONT = "Liberation Sans"


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def configure(run_root: Path, out: Path, font: str | None = None):
    """Point the shared style at ``run_root`` and redirect every writer to ``out``."""
    for folder in PROTECTED:
        if _inside(out, folder):
            raise SystemExit(f"Refusing to write figures into {folder}")
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(key, "2")
    os.environ["BITFI_RUN_ROOT"] = str(run_root)
    os.environ["BITFI_CLEAN"] = "1"
    os.environ["BITFI_RESULT_ROOT"] = str(run_root)
    sys.path[:0] = [str(HERE), str(HERE.parent)]
    import prism_style as ps
    if ps.RESULT_ROOT.resolve() != run_root.resolve():
        raise SystemExit(f"prism_style already bound to {ps.RESULT_ROOT}")
    ps.OUT = out
    import build_prism_figures as b
    b.OUT = out
    b.DATA = out / "source_data"
    b.DATA.mkdir(parents=True, exist_ok=True)
    setup = ps.setup

    def styled_setup():
        setup()
        if font:
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = [font]
    ps.setup = styled_setup
    return ps, b


class Recorder:
    """Replacement for ``prism_style.save`` that also keeps a PNG preview and the
    axis limits, tick labels and text annotations of every saved figure."""

    def __init__(self, ps, preview: Path | None, log_path: Path, positions: bool = False):
        self.ps = ps
        self.preview = preview
        self.log_path = log_path
        # ``positions`` (final renders only) also records each axes box in inches after a draw.
        self.positions = positions
        # Keep the records of figures rendered by earlier runs into the same folder.
        self.records = json.loads(log_path.read_text()) if log_path.exists() else {}

    @staticmethod
    def describe(fig, positions=False):
        axes = []
        if positions:
            fig.canvas.draw()  # applies fixed aspect ratios before the boxes are read
        w, h = fig.get_size_inches()
        for ax in fig.axes:
            if not ax.axison:
                continue
            record = dict(
                title=ax.get_title(loc="left") or ax.get_title(),
                xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(),
                xlim=[float(v) for v in ax.get_xlim()], ylim=[float(v) for v in ax.get_ylim()],
                xticks=[t.get_text() for t in ax.get_xticklabels()],
                yticks=[t.get_text().replace("\n", " ") for t in ax.get_yticklabels()],
                texts=[t.get_text() for t in ax.texts])
            if positions:
                p = ax.get_position()
                record["box_in"] = [round(float(v), 4) for v in (p.x0 * w, p.y0 * h, p.x1 * w, p.y1 * h)]
            axes.append(record)
        return axes

    def save(self, fig, name, supp=False, folder=None, png=False):
        ps = self.ps
        if PARTIAL["on"] and PARTIAL["dropped"]:
            name = f"{name}_provisional"
        ps.finish_axes(fig)
        folder = folder if folder is not None else (ps.OUT / "Supplementary" if supp else ps.OUT)
        folder.mkdir(parents=True, exist_ok=True)
        self.records[name] = dict(file=str(folder / f"{name}.pdf"), size_in=[float(v) for v in fig.get_size_inches()],
                                  axes=self.describe(fig, self.positions))
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=.08)
        import matplotlib.pyplot as plt
        plt.close(fig)
        targets = []
        if png:
            targets.append((folder / name, 300))
        if self.preview is not None:
            self.preview.mkdir(parents=True, exist_ok=True)
            targets.append((self.preview / name, 150))
        for stem, dpi in targets:
            subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(folder / f"{name}.pdf"), str(stem)],
                           check=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(self.records, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Existing figures (M4 re-rendering)
# ---------------------------------------------------------------------------

def seasonal(ps, b, main, models=None):
    """Fig. S4, copied from build_prism_figures.supplementary (unchanged drawing);
    ``models`` only reorders the rows for an ordering preview."""
    import matplotlib.pyplot as plt
    models = list(models or ps.MAIN)
    seas = b.site_scores(main[main.group_type == "season"], ["group_value"])
    seas.to_csv(b.DATA / "season_greenhouse_scores.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    fig.subplots_adjust(left=.32, right=.98, wspace=.27, hspace=.45, bottom=.10, top=.92)
    for i, season in enumerate(["spring", "summer", "fall", "winter"]):
        ax = axes.flat[i]
        b.dotplot(ax, seas[seas.group_value == season], models=models, labels=i % 2 == 0, points=False)
        if i % 2 == 0:
            ax.set_yticklabels([ps.LABELS[m].replace(" (", "\n(", 1) for m in models], fontsize=7)
        ps.panel(ax, chr(65 + i), "Autumn" if season == "fall" else season.capitalize())
    ps.save(fig, "FigureS11_Seasonal_performance", supp=True)


def information_sources(ps, b, full, s, title_b=S5B_TITLE_REVISED):
    """Fig. S5, copied from build_prism_figures.supplementary; only the panel B title
    is a parameter. The indoor/outdoor means are also written as source data."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    models = ["TimesFM3.0", "TimesFM3.0-COV", "TimesFM3.0-COV-SPA", "BiTFI-TimesFM3"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fig.subplots_adjust(left=.28, right=.98, wspace=.65, bottom=.20, top=.84)
    b.dotplot(axes[0], s, models, points=True)
    ps.panel(axes[0], "A", "Available information")
    b.rowsummary(s, models).to_csv(b.DATA / "information_sources_A_summary.csv", index=False)
    v = b.site_scores(full[(full.group_type == "all") & full.model.isin(["Spatial-Ridge", "TimesFM3.0-COV-SPA", "BiTFI-TimesFM3"])], ["variable"])
    rows = []
    for m in ["Spatial-Ridge", "TimesFM3.0-COV-SPA", "BiTFI-TimesFM3"]:
        z = v[v.model == m].copy()
        z["type"] = np.where(z.variable.isin(["Tin", "RH", "CO2"]), "Indoor", "Outdoor")
        q = z.groupby(["greenhouse", "type"]).NMAE.mean().reset_index()
        means = [q[q.type == t].NMAE.mean() for t in ["Indoor", "Outdoor"]]
        rows += [dict(model=m, group=t, NMAE=value) for t, value in zip(["Indoor", "Outdoor"], means)]
        axes[1].plot([0, 1], means, marker="o", color=ps.COLORS[m], label=ps.LABELS[m])
    pd.DataFrame(rows).to_csv(b.DATA / "information_sources_B_indoor_outdoor.csv", index=False)
    axes[1].set_xticks([0, 1], ["Indoor", "Outdoor"])
    axes[1].set_ylabel("NMAE")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, [x.replace(" (", "\n(", 1).replace("local + cross-greenhouse covariates", "local + cross-greenhouse\ncovariates") for x in labels], fontsize=6.3, loc="upper right")
    ps.panel(axes[1], "B", title_b)
    ps.save(fig, "FigureS3_Information_sources", supp=True)


def render_existing(args):
    run_root = Path(args.run_root).resolve()
    out = Path(args.out).resolve() if args.out else run_root / "figures"
    ps, b = configure(run_root, out, args.font)
    if args.aligned_labels:
        # Proofreading Y1 (2026-10-20): ps.panel puts the letter at a fixed fraction of the axes
        # height, which leaves it 5.6-10.1 pt below the title baseline on the shorter axes of
        # Figs. 4, 5 and S4-S6. The letter takes the point-based placement of the new figures
        # instead; the title call and every other artist are the same as before.
        ps.panel = lambda ax, letter, title=None: panel_label(ps, ax, letter, title)
    import pandas as pd
    preview = Path(args.preview).resolve() if args.preview else out / "revision2/preview_existing"
    rec = Recorder(ps, preview, Path(args.log).resolve() if args.log else out / "revision2/render_log_existing.json")
    ps.save = rec.save
    ps.setup()
    an = run_root / "analysis"
    full = pd.read_csv(an / "comparison_results.csv")
    main = full[full.model.isin(ps.MAIN)]
    order = list(ps.MAIN)
    if args.order == "nmae":
        # Ordering preview: decreasing overall NMAE, the order of Tables 3, 4 and S8.
        overall = b.site_scores(main[main.group_type == "all"]).groupby("model").NMAE.mean()
        order = overall.reindex(ps.MAIN).sort_values(ascending=False).index.tolist()
        assert order[-1] == "BiTFI-TimesFM3", order  # Fig. 3B pairs every row with the last one
        b.FIGURE3_MODELS = order
        b.FIGURE45_MODELS = order
        (out / "model_order.json").write_text(json.dumps(dict(order=order, NMAE=overall.reindex(order).tolist()), indent=2))
    only = set(args.only or ["fig3", "fig4", "fig5", "s1", "s4", "s5"])
    if "fig3" in only:
        b.overall(full)
    if "fig4" in only:
        b.gap_robustness(full)
    if "fig5" in only:
        b.variables(full)
    if "s1" in only or "s5" in only:
        s = b.extended_comparison(full) if "s1" in only else b.site_scores(full[full.group_type == "all"])
    if "s4" in only:
        seasonal(ps, b, main, order)
    if "s5" in only:
        information_sources(ps, b, full, s, S5B_TITLE_PUBLISHED if args.published_s5b_title else S5B_TITLE_REVISED)
    if args.with_s6:
        import FigureS7_univariate_backbones as s6
        s6.OUT = out
        s6.main()
    if not args.no_copy_analysis:
        # Same source-table copy as hourly_render.main, into this output folder only.
        for p in an.glob("*.csv"):
            if not p.name.startswith(("early_", "additional_")):
                shutil.copy2(p, b.DATA / p.name)
        for p in an.glob("*.json"):
            shutil.copy2(p, b.DATA / p.name)
    print("Rendered", ", ".join(rec.records), "->", out, flush=True)


# ---------------------------------------------------------------------------
# Comparison of two renders
# ---------------------------------------------------------------------------

def compare(args):
    import numpy as np
    import pandas as pd
    old, new = Path(args.old), Path(args.new)
    lines = [f"# Re-rendered figures: {old} -> {new}", ""]
    od, nd = old / "source_data", new / "source_data"

    def table(name, keys, value):
        a = pd.read_csv(od / name)
        c = pd.read_csv(nd / name)
        m = a.merge(c, on=keys, suffixes=("_old", "_new"), how="outer", indicator=True)
        m["delta"] = m[value + "_new"] - m[value + "_old"]
        return m

    m = table("main_summary.csv", ["model"], "NMAE")
    lines += ["## Fig. 3A main_summary.csv (changed rows)", m.loc[m.delta.abs() > 1e-12, ["model", "NMAE_old", "NMAE_new", "delta", "CI_low_old", "CI_low_new", "CI_high_old", "CI_high_new"]].to_string(index=False), ""]
    m = table("paired_greenhouse_tests.csv", ["reference"], "reduction_percent")
    lines += ["## Fig. 3B paired_greenhouse_tests.csv (changed rows)", m.loc[m.delta.abs() > 1e-12, ["reference", "reduction_percent_old", "reduction_percent_new", "CI_low_old", "CI_low_new", "CI_high_old", "CI_high_new", "holm_p_old", "holm_p_new"]].to_string(index=False), ""]
    a = pd.read_csv(od / "all_model_summary.csv").reset_index().rename(columns={"index": "row"})
    c = pd.read_csv(nd / "all_model_summary.csv").reset_index().rename(columns={"index": "row"})
    m = a.merge(c, on="model", suffixes=("_old", "_new"))
    m = m[(m.row_old != m.row_new) | ((m.NMAE_new - m.NMAE_old).abs() > 1e-12)]
    lines += ["## Fig. S1 order (row 0 = top, NMAE descending) and NMAE", m[["model", "row_old", "row_new", "NMAE_old", "NMAE_new"]].to_string(index=False), ""]
    for name, keys, value in [("information_sources_A_summary.csv", ["model"], "NMAE"),
                              ("univariate_backbone_summary.csv", ["model"], "NMAE"),
                              ("univariate_backbone_paired_tests.csv", ["reference"], "TFM3_reduction_percent")]:
        if (od / name).exists() and (nd / name).exists():
            m = table(name, keys, value)
            cols = [c for c in m.columns if c.endswith(("_old", "_new")) and not c.startswith(("label", "SD", "n_"))]
            lines += [f"## {name} (changed rows)", m.loc[m.delta.abs() > 1e-12, keys + cols + ["delta"]].to_string(index=False), ""]
    for name, keys in [("gap_scenario_greenhouse_scores.csv", ["model", "greenhouse", "scenario", "gap_length_h"]),
                       ("variable_greenhouse_scores.csv", ["model", "greenhouse", "variable"]),
                       ("season_greenhouse_scores.csv", ["model", "greenhouse", "group_value"])]:
        m = table(name, keys, "NMAE")
        changed = m[m.delta.abs() > 1e-12]
        lines += [f"## {name}: {len(changed)} changed greenhouse cells; models {sorted(changed.model.unique())}", ""]
        if len(changed):
            extra = keys[2:]
            g = changed.groupby(["model", *extra]).agg(n=("delta", "size"), max_abs_delta=("delta", lambda x: float(np.abs(x).max()))).reset_index()
            lines += [g.to_string(index=False), ""]
    lo = json.loads(Path(args.old_log).read_text())
    ln = json.loads(Path(args.new_log).read_text())
    lines += ["## Axis limits and tick labels that differ", ""]
    for name in sorted(set(lo) | set(ln)):
        if name not in lo or name not in ln:
            lines.append(f"- {name}: present only in {'new' if name in ln else 'old'}")
            continue
        for i, (x, y) in enumerate(zip(lo[name]["axes"], ln[name]["axes"])):
            for key in ["title", "xlim", "ylim", "xticks", "yticks", "texts"]:
                if key in ("xlim", "ylim"):
                    same = np.allclose(x[key], y[key], rtol=0, atol=1e-9)
                else:
                    same = x[key] == y[key]
                if not same:
                    lines.append(f"- {name} axes[{i}] ({y['title']}) {key}: {x[key]} -> {y[key]}")
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# New revision-2 figures
# ---------------------------------------------------------------------------

MAIN_MODEL = "BiTFI-TimesFM3"
# Row labels follow the manuscript vocabulary: "local" = other sensors of the target
# greenhouse plus calendar features; "cross" = same-variable reference series.
INFO_ROWS = [("BiTFI-calendar", "Calendar only"), ("BiTFI-cross", "Calendar + cross"),
             ("BiTFI-local", "Local"), (MAIN_MODEL, "Local + cross\n(BiTFI)")]
REF_ROWS = [("BiTFI-local", "No reference"), ("BiTFI-k1", "1 reference"), (MAIN_MODEL, "3 references\n(BiTFI)"),
            ("BiTFI-random3", "3 random\nreferences"), ("BiTFI-exclTop5", "3 references,\n5 most used excluded")]
CONTROL_MODELS = ["CovReg-Ridge", "CovReg-LGBM", "TimesFM3.0-COV-SPA", MAIN_MODEL]
CONTROL_MARKERS = {"CovReg-Ridge": "s", "CovReg-LGBM": "^", "TimesFM3.0-COV-SPA": "X", MAIN_MODEL: "o"}
CONTROL_LABELS = {"TimesFM3.0-COV-SPA": "TimesFM3 (local + cross)", MAIN_MODEL: "BiTFI"}
GAPS = [6, 12, 24, 72, 168]
SCENARIO_TITLES = {"A": "Single sensor", "B": "Indoor sensor group", "C": "All sensors"}
ALIASES: dict[str, str] = {}  # stand-in models for layout tests (scratch output only)


def _pd():
    import numpy as np
    import pandas as pd
    return np, pd


PARTIAL = {"on": False, "dropped": []}


def finished(run_root: Path, model: str) -> bool:
    """A configuration counts as finished once its merge/complete marker is written
    (hourly_analysis.save_merged writes merged_complete.json after results.csv)."""
    folder = run_root / "evaluation" / ALIASES.get(model, model)
    if not (folder / "results.csv").exists():
        return False
    if model.startswith("BiTFI-") and model not in ("BiTFI-TimesFM3", "BiTFI-TimesFM3-fwd", "BiTFI-Chronos2") and not ALIASES:
        return (folder / "merged_complete.json").exists()
    if model.startswith("CovReg-") and not ALIASES:
        return (folder / "complete.json").exists()
    return True


def keep_rows(run_root, rows):
    """All rows, or (partial mode) only rows whose configuration has finished."""
    if not PARTIAL["on"]:
        return rows
    kept = [r for r in rows if finished(run_root, r[0] if isinstance(r, tuple) else r)]
    PARTIAL["dropped"] += [r[0] if isinstance(r, tuple) else r for r in rows if r not in kept]
    return kept


def load_eval(run_root: Path, models, reference=MAIN_MODEL):
    """``group_type == 'all'`` rows of ``models``; every configuration must cover the same
    variable-cases with the same evaluated hours as ``reference``."""
    np, pd = _pd()
    frames = []
    for m in models:
        src = ALIASES.get(m, m)
        d = pd.read_csv(run_root / "evaluation" / src / "results.csv")
        d = d[(d.model == src) & (d.group_type == "all")].copy()
        d["model"] = m
        frames.append(d)
    full = pd.concat(frames, ignore_index=True)
    ref = full[full.model == reference].set_index(["case_id", "variable"]).n_eval.sort_index()
    if ref.empty:
        raise ValueError(f"Reference {reference} missing")
    for m in models:
        got = full[full.model == m].set_index(["case_id", "variable"]).n_eval.sort_index()
        pd.testing.assert_series_equal(got, ref, check_names=False)
        if not np.isfinite(full.loc[full.model == m, "NMAE"]).all():
            raise ValueError(f"Non-finite NMAE in {m}")
    return full


def panel_label(ps, ax, letter, title=None):
    """Panel letter on the same baseline as the left-aligned title, as in Fig. 3.

    ``ps.panel`` places the letter at a fixed fraction of the axes height, so on short axes
    it falls below the title. Offsets here are in points and do not depend on axes size."""
    if title:
        ax.set_title(title, loc="left", pad=10)
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-8, 10), textcoords="offset points",
                ha="right", va="baseline", fontweight="bold", fontsize=13, annotation_clip=False)


def dot_panel(ps, ax, values, rows, fmt="{:.4f}", xlabel="NMAE", xmax=None, annotate=True, points=True):
    """Fig. 3A dot plot for labelled rows. ``values`` maps a configuration to its greenhouse
    scores ordered by greenhouse; open points are greenhouses, the diamond is the mean and
    the bar the greenhouse bootstrap 95% CI (``ps.interval``). Returns the plotted summary."""
    np, pd = _pd()
    rng = np.random.default_rng(14)
    right_edge, out = 0., []
    for i, (m, label) in enumerate(rows):
        x = np.asarray(values[m], float)
        mu, lo, hi = ps.interval(x)
        c = ps.COLORS[m]
        if points:
            ax.scatter(x, i + rng.uniform(-.16, .16, len(x)), s=17, facecolors="white", edgecolors=c, lw=.9, alpha=.85, zorder=2)
        ax.errorbar(mu, i, xerr=[[mu - lo], [hi - mu]], fmt="D", ms=4.5, color=c, elinewidth=1.65, capsize=3,
                    capthick=1.2, zorder=3)
        right = max(hi, x.max()) if points else hi
        right_edge = max(right_edge, right)
        if annotate:
            ax.annotate(fmt.format(mu), (right, i), xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=7, color="#303940", annotation_clip=False)
        out.append(dict(model=m, row_label=label.replace("\n", " "), mean=mu, CI_low=lo, CI_high=hi,
                        SD=float(np.std(x, ddof=1)), n_greenhouses=len(x)))
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    ax.set_ylim(len(rows) - .5, -.5)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, xmax if xmax else right_edge * 1.32)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
    ax.tick_params(axis="y", length=0, pad=6, labelsize=7.2)
    ax.spines["left"].set_visible(False)
    return pd.DataFrame(out)


def reduction_panel(ps, ax, table, rows, xlabel="Error reduction with BiTFI (%)", n_slots=None):
    """Fig. 3B form: BiTFI reduction against each reference with its paired 95% CI.

    ``n_slots`` is the row count of the neighbouring dot panel; using it for the y limits keeps
    each reference on the same horizontal line as its row there (the BiTFI slot stays empty)."""
    lows, highs = [], []
    for i, (m, _) in enumerate(rows):
        r = table.loc[m]
        ax.errorbar(r.reduction, i, xerr=[[r.reduction - r.low], [r.high - r.reduction]], fmt="o", color=ps.COLORS[m],
                    capsize=3, ms=4.5)
        ax.annotate(f"{r.reduction:.1f}%", (r.high, i), xytext=(5, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=7, color="#303940", annotation_clip=False)
        lows.append(r.low); highs.append(r.high)
    ax.set_yticks(range(len(rows)), ["" for _ in rows])
    n = n_slots or len(rows)
    ax.set_ylim(n - .5, -.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(min(0, min(lows)) - 3, max(highs) * 1.28)
    ax.axvline(0, color="#BBBBBB", lw=.8, zorder=0)
    ax.set_xlabel(xlabel)


def greenhouse_values(scores, models, column="NMAE"):
    return {m: scores[scores.model == m].sort_values("greenhouse")[column].to_numpy() for m in models}


def gap_lines(ps, ax, scores, models, labels, offsets=.1):
    """Fig. 4 line form with greenhouse bootstrap 95% CIs (S6 error-bar style)."""
    np, pd = _pd()
    centre = (len(models) - 1) / 2
    out = []
    for j, m in enumerate(models):
        mus, low, high = [], [], []
        for g in GAPS:
            mu, lo, hi = ps.interval(scores[(scores.model == m) & (scores.gap_length_h == g)].sort_values("greenhouse").NMAE.to_numpy())
            mus.append(mu); low.append(mu - lo); high.append(hi - mu)
            out.append(dict(model=m, gap_length_h=g, mean=mu, CI_low=lo, CI_high=hi))
        ax.errorbar(np.arange(len(GAPS)) + (j - centre) * offsets, mus, yerr=[low, high], color=ps.COLORS[m],
                    marker=CONTROL_MARKERS.get(m, "o"), ms=3.8, lw=2 if m == MAIN_MODEL else 1.2, capsize=2,
                    elinewidth=.9, label=labels.get(m, ps.LABELS[m]), zorder=3 if m == MAIN_MODEL else 2)
    ax.set_xticks(range(len(GAPS)), [str(g) for g in GAPS])
    ax.set_xlabel("Gap length (h)")
    ax.set_ylim(bottom=0)
    return pd.DataFrame(out)


def crosscheck(summary, path, key="model", mean="mean", low="low", high="high", where=None):
    """Assert that the plotted means and CIs equal the analysis table of the experiment."""
    _, pd = _pd()
    if not path.exists():
        return f"{path.name}: not available"
    t = pd.read_csv(path)
    if where is not None:
        t = where(t)
    t = t.set_index(key)
    worst = 0.
    for r in summary.itertuples():
        if r.model not in t.index:
            continue
        q = t.loc[r.model]
        worst = max(worst, abs(q[mean] - r.mean), abs(q[low] - r.CI_low), abs(q[high] - r.CI_high))
    if worst > 1e-12:
        raise ValueError(f"Plotted summary differs from {path} by {worst}")
    return f"{path.name}: identical (max abs difference {worst:.1e})"


def fig7_inputs(run_root):
    models = sorted({m for m, _ in INFO_ROWS + REF_ROWS} | set(CONTROL_MODELS))
    return [run_root / "evaluation" / ALIASES.get(m, m) / "results.csv" for m in models]


def build_fig7(ps, b, run_root, folder, rec):
    """Information sources within BiTFI (A), reference number and selection (B) and the
    equal-information regression controls by gap length, one panel per scenario (C-E)."""
    np, pd = _pd()
    import matplotlib.pyplot as plt
    data = folder / "source_data"
    data.mkdir(parents=True, exist_ok=True)
    info_rows, ref_rows = keep_rows(run_root, INFO_ROWS), keep_rows(run_root, REF_ROWS)
    controls = keep_rows(run_root, CONTROL_MODELS)
    models = list(dict.fromkeys([m for m, _ in info_rows + ref_rows] + controls))
    full = load_eval(run_root, models)
    s = b.site_scores(full)
    s.to_csv(data / "Figure7_greenhouse_scores.csv", index=False)
    g = b.site_scores(full[full.model.isin(controls)], ["scenario", "gap_length_h"])
    g.to_csv(data / "Figure7_gap_scenario_greenhouse_scores.csv", index=False)
    values = greenhouse_values(s, models)
    dotted = [m for m, _ in info_rows + ref_rows]
    xmax = max(max(values[m].max(), ps.interval(values[m])[2]) for m in dotted) * 1.32
    fig = plt.figure(figsize=(7.2, 6.5))
    ax_a = fig.add_axes([.20, .615, .29, .32])
    ax_b = fig.add_axes([.685, .615, .29, .32])
    sa = dot_panel(ps, ax_a, values, info_rows, xmax=xmax)
    panel_label(ps, ax_a, "A", "Information sources")
    sb = dot_panel(ps, ax_b, values, ref_rows, xmax=xmax)
    panel_label(ps, ax_b, "B", "Reference number and choice")
    axes = [fig.add_axes([.085 + j * .305, .085, .265, .30]) for j in range(3)]
    lines = []
    for j, sc in enumerate(["A", "B", "C"]):
        lines.append(gap_lines(ps, axes[j], g[g.scenario == sc], controls, CONTROL_LABELS).assign(scenario=sc))
        panel_label(ps, axes[j], chr(67 + j), SCENARIO_TITLES[sc])
    top_y = max(ax.get_ylim()[1] for ax in axes)
    for j, ax in enumerate(axes):
        ax.set_ylim(0, top_y)
        if j:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("NMAE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="center", bbox_to_anchor=(.53, .478), fontsize=7.5,
               handlelength=1.7, columnspacing=1.2)
    summary = pd.concat([sa.assign(panel="A"), sb.assign(panel="B")], ignore_index=True)
    summary.to_csv(data / "Figure7_AB_summary.csv", index=False)
    pd.concat(lines, ignore_index=True).to_csv(data / "Figure7_CDE_gap_summary.csv", index=False)
    an = run_root / "analysis/revision2"
    checks = [crosscheck(summary.drop_duplicates("model"), an / "bitfi_variants_summary.csv"),
              crosscheck(b.rowsummary(s, controls).rename(columns={"NMAE": "mean"}), an / "covreg_summary.csv")]
    for name in ["bitfi_variants_paired.csv", "covreg_paired.csv", "covreg_scenario_gap.csv", "bitfi_variants_scenarios.csv"]:
        if (an / name).exists():
            shutil.copy2(an / name, data / f"Figure7_{name}")
    (data / "Figure7_crosscheck.txt").write_text("\n".join(checks) + "\n")
    rec.save(fig, "Figure7_Information_sources", folder=folder, png=True)


# Mathematical text (units) in the new figures uses the same Liberation Sans as the rest
# of the figure, so no DejaVu glyphs are embedded.
MATH_RC = {"mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
           "mathtext.bf": "Liberation Sans:bold", "mathtext.sf": "Liberation Sans", "mathtext.default": "regular"}
INDICATORS = [  # (file key, metric, scope, panel title, x label, number format)
    ("vpd", "VPD_MAE", "B+C", "Vapor pressure deficit", "Hourly VPD MAE (kPa)", "{:.3f}"),
    ("radiation", "H_AE", "A+C", "Daily solar radiation", "Daily radiation sum error (MJ m$^{-2}$ d$^{-1}$)", "{:.2f}"),
]


def representative_rows(ps):
    return [(m, ps.LABELS[m].replace(" (", "\n(", 1)) for m in ps.MAIN]


def fig8_inputs(run_root):
    an = run_root / "analysis/revision2"
    return [an / f"indicators_{k}_{t}.csv" for k, *_ in INDICATORS for t in ["sites", "summary", "paired"]]


def build_fig8(ps, b, run_root, folder, rec):
    """Derived agricultural indicators for the eight representative configurations, in the
    two-column form of Fig. 3: absolute error (A, C) and BiTFI's paired reduction (B, D)."""
    np, pd = _pd()
    import matplotlib.pyplot as plt
    an = run_root / "analysis/revision2"
    data = folder / "source_data"
    data.mkdir(parents=True, exist_ok=True)
    rows = representative_rows(ps)
    with plt.rc_context(MATH_RC):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.4), gridspec_kw={"width_ratios": [1.05, 1]})
        fig.subplots_adjust(left=.22, right=.98, wspace=.48, hspace=.42, bottom=.08, top=.94)
        checks, plotted, reductions = [], [], []
        for r, (key, metric, scope, title, xlabel, fmt) in enumerate(INDICATORS):
            pick = lambda t: t[(t.metric == metric) & (t.scope == scope) & (t.variant == "primary")]
            sites = pick(pd.read_csv(an / f"indicators_{key}_sites.csv"))
            values = {m: sites[sites.model == m].sort_values("greenhouse").value.to_numpy() for m, _ in rows}
            summ = dot_panel(ps, axes[r, 0], values, rows, fmt=fmt, xlabel=xlabel)
            axes[r, 0].tick_params(axis="y", labelsize=7)
            checks.append(crosscheck(summ, an / f"indicators_{key}_summary.csv", where=pick))
            paired = pick(pd.read_csv(an / f"indicators_{key}_paired.csv"))
            paired = paired[paired.family == "representative_holm7"].set_index("reference")
            reduction_panel(ps, axes[r, 1], paired, rows[:-1], n_slots=len(rows))
            panel_label(ps, axes[r, 0], chr(65 + 2 * r), title)
            panel_label(ps, axes[r, 1], chr(66 + 2 * r), "BiTFI improvement")
            plotted.append(summ.assign(indicator=key, metric=metric, scope=scope))
            reductions.append(paired.reset_index().assign(indicator=key))
        pd.concat(plotted).to_csv(data / "Figure8_indicator_summary.csv", index=False)
        pd.concat(reductions).to_csv(data / "Figure8_indicator_paired.csv", index=False)
        (data / "Figure8_crosscheck.txt").write_text("\n".join(checks) + "\n")
        rec.save(fig, "Figure8_Agricultural_indicators", folder=folder, png=True)


def fig7c_inputs(run_root):
    return fig7_inputs(run_root) + fig8_inputs(run_root)


def build_fig7_combined(ps, b, run_root, folder, rec):
    """Single-figure alternative: A and B as in Fig. 7, C the equal-information controls
    pooled over scenarios by gap length, D BiTFI's paired reductions of the two derived
    indicators (both in percent, one axis)."""
    np, pd = _pd()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    an = run_root / "analysis/revision2"
    data = folder / "source_data"
    data.mkdir(parents=True, exist_ok=True)
    info_rows, ref_rows = keep_rows(run_root, INFO_ROWS), keep_rows(run_root, REF_ROWS)
    controls = keep_rows(run_root, CONTROL_MODELS)
    models = list(dict.fromkeys([m for m, _ in info_rows + ref_rows] + controls))
    full = load_eval(run_root, models)
    s = b.site_scores(full)
    values = greenhouse_values(s, models)
    dotted = [m for m, _ in info_rows + ref_rows]
    xmax = max(max(values[m].max(), ps.interval(values[m])[2]) for m in dotted) * 1.32
    with plt.rc_context(MATH_RC):
        fig = plt.figure(figsize=(7.2, 6.6))
        ax_a = fig.add_axes([.20, .60, .29, .335])
        ax_b = fig.add_axes([.685, .60, .29, .335])
        ax_c = fig.add_axes([.085, .085, .36, .36])
        ax_d = fig.add_axes([.685, .085, .29, .36])
        dot_panel(ps, ax_a, values, info_rows, xmax=xmax)
        panel_label(ps, ax_a, "A", "Information sources")
        dot_panel(ps, ax_b, values, ref_rows, xmax=xmax)
        panel_label(ps, ax_b, "B", "Reference number and choice")
        pooled = b.site_scores(full[full.model.isin(controls)], ["gap_length_h"])
        lines = gap_lines(ps, ax_c, pooled, controls, CONTROL_LABELS)
        ax_c.set_ylabel("NMAE")
        ax_c.legend(fontsize=6.8, loc="upper left", handlelength=1.6, ncol=1, borderaxespad=0)
        ax_c.set_ylim(0, ax_c.get_ylim()[1] * 1.25)
        panel_label(ps, ax_c, "C", "Equal-information controls")
        rows = representative_rows(ps)[:-1]
        markers = {"vpd": ("o", "Hourly VPD"), "radiation": ("s", "Daily radiation sum")}
        kept = []
        for j, (key, metric, scope, *_rest) in enumerate(INDICATORS):
            pick = lambda t: t[(t.metric == metric) & (t.scope == scope) & (t.variant == "primary")]
            paired = pick(pd.read_csv(an / f"indicators_{key}_paired.csv"))
            paired = paired[paired.family == "representative_holm7"].set_index("reference")
            for i, (m, _) in enumerate(rows):
                r = paired.loc[m]
                y = i + (j - .5) * .3
                ax_d.errorbar(r.reduction, y, xerr=[[r.reduction - r.low], [r.high - r.reduction]], fmt=markers[key][0],
                              color=ps.COLORS[m], mfc="white" if key == "radiation" else ps.COLORS[m], capsize=2, ms=4, lw=1.1)
            kept.append(paired.reset_index().assign(indicator=key))
        ax_d.set_yticks(range(len(rows)), [label for _, label in rows])
        ax_d.tick_params(axis="y", length=0, labelsize=7)
        ax_d.set_ylim(len(rows) - .5, -.5)
        ax_d.spines["left"].set_visible(False)
        ax_d.axvline(0, color="#BBBBBB", lw=.8, zorder=0)
        ax_d.set_xlabel("Error reduction with BiTFI (%)")
        ax_d.set_xlim(left=min(0, min(k.low.min() for k in kept)) - 3)
        ax_d.legend(handles=[Line2D([0], [0], marker=markers[k][0], color="#59636C", mfc="#59636C" if k == "vpd" else "white",
                                    lw=0, ms=4, label=markers[k][1]) for k in markers],
                    fontsize=6.8, loc="lower right", handletextpad=.3, borderaxespad=0)
        panel_label(ps, ax_d, "D", "Derived indicators")
        lines.to_csv(data / "Figure7_combined_C_gap_summary.csv", index=False)
        pd.concat(kept).to_csv(data / "Figure7_combined_D_indicator_paired.csv", index=False)
        notes = []
        if (an / "covreg_gap.csv").exists():
            ref = pd.read_csv(an / "covreg_gap.csv").merge(lines, on=["model", "gap_length_h"], suffixes=("", "_plot"))
            worst = float(np.max(np.abs(ref[["mean", "low", "high"]].to_numpy() - ref[["mean_plot", "CI_low", "CI_high"]].to_numpy()))) if len(ref) else 0.
            if worst > 1e-12:
                raise ValueError(f"Panel C differs from covreg_gap.csv by {worst}")
            notes.append(f"covreg_gap.csv: {len(ref)} model-gap cells identical (max abs difference {worst:.1e})")
        (data / "Figure7_combined_crosscheck.txt").write_text("\n".join(notes) + "\n")
        rec.save(fig, "Figure7_Information_sources_combined", folder=folder / "alternatives", png=True)


DETAIL_ROWS = [("CovReg-Ridge", "Ridge (local + cross)"), ("CovReg-LGBM", "LightGBM (local + cross)"),
               ("TimesFM3.0-COV-SPA", "TimesFM3 (local + cross)"), ("BiTFI-calendar", "BiTFI, calendar only"),
               ("BiTFI-cross", "BiTFI, calendar + cross"), ("BiTFI-local", "BiTFI, local"),
               ("BiTFI-k1", "BiTFI, 1 reference"), ("BiTFI-random3", "BiTFI, 3 random references"),
               ("BiTFI-exclTop5", "BiTFI, 5 most-used excluded"), (MAIN_MODEL, "BiTFI")]


def detail_inputs(run_root):
    return [run_root / "evaluation" / ALIASES.get(m, m) / "results.csv" for m, _ in DETAIL_ROWS]


def _detail(ps, b, run_root, folder, rec, column, levels, titles, name, shape, size, rows=None, group=None):
    """Dot panels of the ten control rows by ``column``. ``rows`` replaces the labels of
    ``DETAIL_ROWS`` and ``group`` = (text, first, last configuration) adds a bracket left of those
    rows in each labelled panel (final render)."""
    np, pd = _pd()
    import matplotlib.pyplot as plt
    data = folder / "source_data"
    data.mkdir(parents=True, exist_ok=True)
    detail_rows = keep_rows(run_root, rows or DETAIL_ROWS)
    models = [m for m, _ in detail_rows]
    full = load_eval(run_root, models)
    s = b.site_scores(full, [column])
    s.to_csv(data / f"{name}_greenhouse_scores.csv", index=False)
    fig, axes = plt.subplots(*shape, figsize=size, squeeze=False)
    fig.subplots_adjust(left=.27, right=.98, wspace=.5, hspace=.42 if shape[0] > 1 else .3,
                        top=.94 if shape[0] > 1 else .86, bottom=.07 if shape[0] > 1 else .14)
    out = []
    for i, level in enumerate(levels):
        ax = axes.flat[i]
        z = s[s[column] == level]
        values = greenhouse_values(z, models)
        rows = detail_rows if i % shape[1] == 0 else [(m, "") for m, _ in detail_rows]
        # Greenhouse points are omitted, as in Fig. 5 and Fig. S4.
        q = dot_panel(ps, ax, values, rows, annotate=False, points=False)
        ax.set_xlim(0, max(ps.interval(v)[2] for v in values.values()) * 1.12)
        ax.tick_params(axis="y", labelsize=7)
        panel_label(ps, ax, chr(65 + i), titles[level])
        if group is not None and i % shape[1] == 0:
            group_bracket(ax, models.index(group[1]), models.index(group[2]), group[0])
        out.append(q.assign(**{column: level}))
    for ax in axes.flat[len(levels):]:
        ax.axis("off")
    summary = pd.concat(out, ignore_index=True)
    summary.to_csv(data / f"{name}_summary.csv", index=False)
    an = run_root / "analysis/revision2"
    checks = []
    for file in ([f"bitfi_variants_{'scenarios' if column == 'scenario' else 'variables'}.csv",
                  "covreg_scenario.csv" if column == "scenario" else "covreg_variables_NMAE.csv"]):
        path = an / file
        if path.exists():
            t = pd.read_csv(path)
            worst = 0.
            for r in summary.itertuples():
                hit = t[(t.model == r.model) & (t[column] == getattr(r, column))]
                if len(hit):
                    h = hit.iloc[0]
                    worst = max(worst, abs(h["mean"] - r.mean), abs(h.low - r.CI_low), abs(h.high - r.CI_high))
            if worst > 1e-12:
                raise ValueError(f"{name} differs from {file} by {worst}")
            checks.append(f"{file}: identical (max abs difference {worst:.1e})")
        else:
            checks.append(f"{file}: not available")
    (data / f"{name}_crosscheck.txt").write_text("\n".join(checks) + "\n")
    rec.save(fig, name, folder=folder / "Supplementary", png=True)


def build_detail_scenario(ps, b, run_root, folder, rec):
    _detail(ps, b, run_root, folder, rec, "scenario", ["A", "B", "C"], SCENARIO_TITLES,
            "FigureS13_Controls_by_scenario", (1, 3), (7.2, 4.2))


def build_detail_variable(ps, b, run_root, folder, rec, rows=None, group=None):
    _detail(ps, b, run_root, folder, rec, "variable", ps.VARS, ps.VL,
            "FigureS15_Controls_by_variable", (2, 3), (7.2, 7.4), rows=rows, group=group)


def saits_inputs(run_root):
    an = run_root / "analysis/revision2"
    return [an / f"saits_seeds_{t}.csv" for t in ["summary", "site_scores", "paired", "spread"]]


def build_saits_seeds(ps, b, run_root, folder, rec):
    """SAITS training-seed repeats: per-seed NMAE (A), BiTFI's paired reduction against
    SAITS (local + cross) (B) and the reference gain within SAITS (C)."""
    np, pd = _pd()
    import matplotlib.pyplot as plt
    an = run_root / "analysis/revision2"
    data = folder / "source_data"
    data.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(an / "saits_seeds_summary.csv")
    runs = runs[runs.role != "reproduction (reference only)"]
    site = pd.read_csv(an / "saits_seeds_site_scores.csv")
    paired = pd.read_csv(an / "saits_seeds_paired.csv")
    paired["seed"] = paired.seed.astype(str)
    seeds = sorted(int(x) for x in runs.seed.unique())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3))
    fig.subplots_adjust(left=.085, right=.99, bottom=.19, top=.78, wspace=.45)
    ax = axes[0]
    bit = site[site.model == MAIN_MODEL].sort_values("greenhouse").NMAE.to_numpy()
    mu, lo, hi = ps.interval(bit)
    ax.axhspan(lo, hi, color=ps.COLORS[MAIN_MODEL], alpha=.12, lw=0, zorder=0)
    ax.axhline(mu, color=ps.COLORS[MAIN_MODEL], lw=1.2, zorder=1, label="BiTFI")
    rows = []
    for j, model in enumerate(["SAITS-matched-local", "SAITS-spatial"]):
        z = runs[runs.base_model == model].set_index("seed")
        ys = np.array([z.loc[s, "NMAE"] for s in seeds])
        lows = np.array([z.loc[s, "NMAE_low"] for s in seeds])
        highs = np.array([z.loc[s, "NMAE_high"] for s in seeds])
        mean = site[site.model == f"{model}_seedmean"].sort_values("greenhouse").NMAE.to_numpy()
        m2, l2, h2 = ps.interval(mean)
        xs = np.arange(len(seeds) + 1) + (j - .5) * .2
        ax.errorbar(xs[:-1], ys, yerr=[ys - lows, highs - ys], fmt="o", color=ps.COLORS[model], capsize=2, ms=3.8,
                    lw=1.1, label=ps.LABELS[model])
        ax.errorbar(xs[-1:], [m2], yerr=[[m2 - l2], [h2 - m2]], fmt="D", color=ps.COLORS[model], capsize=3, ms=4.5, lw=1.4)
        rows += [dict(model=model, seed=s, NMAE=y, CI_low=a, CI_high=c) for s, y, a, c in zip(seeds, ys, lows, highs)]
        rows.append(dict(model=model, seed="seed mean", NMAE=m2, CI_low=l2, CI_high=h2))
    rows.append(dict(model=MAIN_MODEL, seed="", NMAE=mu, CI_low=lo, CI_high=hi))
    ax.set_xticks(range(len(seeds) + 1), [str(s) for s in seeds] + ["Mean"])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("NMAE")
    ax.set_ylim(0, max(r["CI_high"] for r in rows) * 1.12)
    ax.set_xlim(-.5, len(seeds) + .5)
    panel_label(ps, ax, "A", "Test error")
    h, l = ax.get_legend_handles_labels()
    order = [1, 2, 0]
    fig.legend([h[i] for i in order], [l[i] for i in order], ncol=3, loc="upper center", bbox_to_anchor=(.53, 1.0), fontsize=7.5)
    panels = [("BiTFI reduction vs SAITS-spatial", "B", "BiTFI improvement", "Error reduction (%)", "SAITS-spatial"),
              ("reference gain: SAITS-spatial vs SAITS-matched-local", "C", "Reference gain in SAITS", "Error reduction (%)", "SAITS-spatial")]
    kept = []
    for kind, letter, title, ylabel, colour in panels:
        ax = axes[ord(letter) - 65]
        q = paired[paired.comparison == kind]
        per = q[q.seed.isin([str(s) for s in seeds])].set_index("seed")
        both = q[q.seed == "greenhouse+seed bootstrap"].iloc[0]
        ys = np.array([per.loc[str(s), "reduction"] for s in seeds])
        lows = np.array([per.loc[str(s), "low"] for s in seeds])
        highs = np.array([per.loc[str(s), "high"] for s in seeds])
        ax.errorbar(range(len(seeds)), ys, yerr=[ys - lows, highs - ys], fmt="o", color=ps.COLORS[colour], capsize=2,
                    ms=3.8, lw=1.1)
        ax.errorbar([len(seeds)], [both.reduction], yerr=[[both.reduction - both.low], [both.high - both.reduction]],
                    fmt="D", color=ps.COLORS[colour], capsize=3, ms=4.5, lw=1.4)
        ax.axhline(0, color="#BBBBBB", lw=.8, zorder=0)
        ax.set_xticks(range(len(seeds) + 1), [str(s) for s in seeds] + ["All"])
        ax.set_xlabel("Training seed")
        ax.set_ylabel(ylabel)
        ax.set_ylim(min(0, lows.min(), both.low) - 2, max(highs.max(), both.high) * 1.12)
        ax.set_xlim(-.5, len(seeds) + .5)
        panel_label(ps, ax, letter, title)
        kept.append(q.assign(panel=letter))
    pd.DataFrame(rows).to_csv(data / "FigureS9_SAITS_seeds_A.csv", index=False)
    pd.concat(kept).to_csv(data / "FigureS9_SAITS_seeds_BC.csv", index=False)
    rec.save(fig, "FigureS9_SAITS_seed_repeats", folder=folder / "Supplementary", png=True)


def loo_inputs(run_root):
    an = run_root / "analysis/revision2"
    return [an / "loo_effect_sizes.csv", an / "loo_effect_sizes_detail.csv"]


def build_loo(ps, b, run_root, folder, rec):
    """Greenhouse dependence of BiTFI's reductions: full-sample estimate with paired bootstrap
    CI and the ten leave-one-greenhouse-out estimates (A); single-greenhouse reductions (B)."""
    np, pd = _pd()
    import matplotlib.pyplot as plt
    an = run_root / "analysis/revision2"
    data = folder / "source_data"
    data.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(an / "loo_effect_sizes.csv")
    detail = pd.read_csv(an / "loo_effect_sizes_detail.csv")
    table = table[table.family == "representative_holm7"].set_index("reference")
    refs = [m for m in ps.MAIN if m != MAIN_MODEL]
    rows = [(m, ps.LABELS[m].replace(" (", "\n(", 1)) for m in refs]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.2, right=.98, bottom=.13, top=.88, wspace=.12)
    rng = np.random.default_rng(14)
    for i, (m, _) in enumerate(rows):
        r = table.loc[m]
        d = detail[(detail.reference == m) & (detail.model == MAIN_MODEL)]
        c = ps.COLORS[m]
        axes[0].scatter(d.reduction, i + rng.uniform(-.14, .14, len(d)), s=12, facecolors="white", edgecolors=c, lw=.8,
                        alpha=.85, zorder=2)
        axes[0].errorbar(r.reduction, i, xerr=[[r.reduction - r.low], [r.high - r.reduction]], fmt="o", color=c,
                         capsize=3, ms=4.5, zorder=3)
        axes[1].scatter(d.excluded_site_reduction, i + rng.uniform(-.14, .14, len(d)), s=17, facecolors="white",
                        edgecolors=c, lw=.9, alpha=.85, zorder=2)
        axes[1].plot([d.excluded_site_reduction.median()] * 2, [i - .25, i + .25], color=c, lw=1.6, zorder=3)
    lo = min(0, detail[detail.model == MAIN_MODEL].excluded_site_reduction.min(), table.loc[refs].low.min())
    hi = max(detail[detail.model == MAIN_MODEL].excluded_site_reduction.max(), table.loc[refs].high.max())
    for ax in axes:
        ax.set_xlim(lo - 3, hi * 1.06)
        ax.set_ylim(len(rows) - .5, -.5)
        ax.axvline(0, color="#BBBBBB", lw=.8, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Error reduction with BiTFI (%)")
    axes[0].set_yticks(range(len(rows)), [label for _, label in rows])
    axes[0].tick_params(axis="y", labelsize=7.2, pad=6)
    axes[1].set_yticks(range(len(rows)), ["" for _ in rows])
    panel_label(ps, axes[0], "A", "Leave one greenhouse out")
    panel_label(ps, axes[1], "B", "Single greenhouses")
    table.reset_index().to_csv(data / "FigureS8_LOO_summary.csv", index=False)
    detail.to_csv(data / "FigureS8_LOO_detail.csv", index=False)
    rec.save(fig, "FigureS8_Greenhouse_dependence", folder=folder / "Supplementary", png=True)


# ---------------------------------------------------------------------------
# Redesign previews (F2, 2026-10-20). The information-source and indicator figures in
# visual forms other than the dot-and-interval rows of Fig. 3 and the gap-length lines of
# Fig. 4, after Moon et al. (2021, Sensors 21, 2187) Figs. 6 (mirrored ablation bars) and 4
# (observed-versus-imputed density grid). Additive: the builders above and their outputs
# are unchanged, and these keys run only when named, e.g. ``new --only fig6r fig7r s15r s11h``.
# Outputs: <out>/revision2_redesign/ (PDF, 300 dpi PNG, source_data/, render log).
# ---------------------------------------------------------------------------

REDESIGN_FOLDER = "revision2_redesign"
INK, MUTED_INK = "#303940", "#68747D"
GREENHOUSE_LINE = "#BFC6CD"
# Author decision (2026-10-20): Fig. 6B draws every greenhouse in the same grey; the one
# greenhouse where the regression was lower is reported in the text, not marked in the figure.
HIGHLIGHT_EXCEPTION = False
HIGHLIGHT = "#2A78D6"  # dataviz validator, all pairs with the three marker colours of Fig. 6r B: normal-vision dE >= 15.8
ABLATION_ROWS = [  # (configuration, label written inside the bar, group)
    (MAIN_MODEL, "Full BiTFI\n(top 3 by |r|)", "full"),
    ("BiTFI-cross", "−Local sensors\n(calendar + cross)", "sources"),
    ("BiTFI-local", "−Cross references\n(local)", "sources"),
    ("BiTFI-calendar", "Calendar only", "sources"),
    ("BiTFI-k1", "Top 1 reference", "references"),
    ("BiTFI-random3", "3 random eligible\nreferences", "references"),
    ("BiTFI-exclTop5", "Top 3 excluding\nthe 5 most used", "references"),
]
ABLATION_GROUPS = [("sources", "Information sources"), ("references", "Reference number and choice")]
SLOPE_ROWS = [("TimesFM3.0-COV-SPA", "TimesFM3\n(local + cross)"), ("CovReg-LGBM", "LightGBM\n(local + cross)"),
              (MAIN_MODEL, "BiTFI")]
# Author decision (2026-10-20): rows run from the trained imputer to BiTFI, top to bottom.
SCATTER_ROWS = [("SAITS-spatial", "SAITS\n(local + cross)"), ("CovReg-LGBM", "LightGBM\n(local + cross)"),
                ("TimesFM3.0-COV-SPA", "TimesFM3\n(local + cross)"), (MAIN_MODEL, "BiTFI")]
# (key, E5 indicator, E5 metric, scope, scenarios, title, unit, axis limits, ticks, hexagons across, MAE format, ramp)
SCATTER_COLUMNS = [
    ("hourly_vpd", "vpd", "VPD_MAE", "B+C", ("B", "C"), "Hourly VPD", "kPa", (-.3, 3.), [0, 1, 2, 3], 55, "{:.4f}", "vpd"),
    ("daily_vpd", "vpd", "VPD_daily_AE", "B+C", ("B", "C"), "Daily mean VPD", "kPa", (-.1, 1.6), [0, .5, 1, 1.5], 26,
     "{:.4f}", "vpd"),
    ("daily_radiation", "radiation", "H_AE", "A+C", ("A", "C"), "Daily radiation sum", "MJ m$^{-2}$ d$^{-1}$", (0., 44.),
     [0, 10, 20, 30, 40], 26, "{:.3f}", "rad"),
]
# Sequential single-hue ramps around the Fig. 1B variable colours (RH #6DA58D for VPD, Rad #C5B66E for
# radiation). dataviz ramp check: monotone lightness, adjacent dL >= 0.06, one hue; the light end (1.5:1 on
# white) is kept pale on purpose so that sparse hexagons recede, as in a density map.
RAMPS = {"vpd": ["#BBD8C6", "#93C0A6", "#6DA58D", "#4C866C", "#33664F", "#1D4636"],
         "rad": ["#DDD2A0", "#CBBB78", "#B4A35A", "#93833F", "#6D602B", "#473F1B"],
         "nmae": ["#E6ECF2", "#C3CFDC", "#9DAFC3", "#7890AB", "#58718F", "#3B5372", "#22344C"]}


def _agree(name, got, expected, tol=1e-12):
    """Largest absolute difference between plotted values and an analysis table; raises above ``tol``."""
    np, _ = _pd()
    got, expected = np.asarray(got, float), np.asarray(expected, float)
    if got.shape != expected.shape:
        raise ValueError(f"{name}: shape {got.shape} versus {expected.shape}")
    worst = float(np.max(np.abs(got - expected))) if got.size else 0.
    if not worst <= tol:
        raise ValueError(f"{name}: plotted values differ from the table by {worst}")
    return f"{name}: identical (max abs difference {worst:.1e}, n={got.size})"


def paired_reduction(x, y):
    """100 (1 - mean x / mean y) and the greenhouse-paired bootstrap 95% CI of
    ``analyze_revision_experiments.compare`` (10,000 draws, default_rng(42))."""
    np, _ = _pd()
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = np.random.default_rng(42).integers(len(x), size=(10000, len(x)))
    lo, hi = np.quantile(100 * (1 - x[i].mean(1) / y[i].mean(1)), [.025, .975])
    return 100 * (1 - x.mean() / y.mean()), lo, hi


def test_ids(run_root):
    """Manuscript names of the test greenhouses (Table 1 order, as numbered for Fig. 6 by
    build_hourly_manuscript.py)."""
    _, pd = _pd()
    coverage = pd.read_csv(run_root / "figures/source_data/Figure1_profile/coverage.csv")
    test = coverage[coverage.role == "test"].reset_index(drop=True)
    return {name: f"Test {i + 1:02d}" for i, name in enumerate(test.name)}


def _one_line(label):
    """Label text without its line breaks; a break after a hyphen ("most-\\nused") closes up."""
    return label.replace("-\n", "-").replace("\n", " ")


def _signed(value, digits=1):
    """Percentage with an explicit sign and a typographic minus (U+2212)."""
    return ("\u2212" if value < 0 else "+") + f"{abs(value):.{digits}f}%"


def _ablation_positions(rows, gap=.35):
    xs, x, last = [], 0., None
    for _, _, group in rows:
        if last is not None:
            x += 1 + (gap if group != last else 0)
        xs.append(x)
        last = group
    return xs


def _outward_left(ax, offset=5.):
    """House spine offset for axes whose bottom spine is replaced by a baseline (finish_axes skips them)."""
    ax.spines["left"].set_position(("outward", offset))
    for side in ["top", "right", "bottom"]:
        ax.spines[side].set_visible(False)


def fig6r_inputs(run_root):
    an = run_root / "analysis/revision2"
    models = [m for m, _, _ in ABLATION_ROWS] + [m for m, _ in SLOPE_ROWS]
    return [run_root / "evaluation" / ALIASES.get(m, m) / "results.csv" for m in dict.fromkeys(models)] + [
        an / "bitfi_variants_summary.csv", an / "bitfi_variants_paired.csv", an / "covreg_sites.csv",
        an / "covreg_paired.csv", an / "covreg_paired_context.csv", run_root / "analysis/all_site_scores.csv",
        run_root / "figures/source_data/Figure1_profile/coverage.csv"]


def build_fig6_redesign(ps, b, run_root, folder, rec, stem="Figure6_Information_sources_preview", prefix="Figure6r",
                        bar_labels=None):
    """New Fig. 6 candidate. (A) Moon Fig. 6 form: overall NMAE of BiTFI and its six covariate
    variants as solid bars (greenhouse bootstrap 95% CI) and, mirrored below, the paired NMAE
    increase over full BiTFI as light bars (paired 95% CI). (B) Greenhouse slope graph from
    TimesFM3 (local + cross) to the equal-information LightGBM regression to BiTFI.

    ``stem`` and ``prefix`` name the figure and its source tables; ``bar_labels`` (configuration ->
    in-bar text) replaces the preview labels of ``ABLATION_ROWS`` (final render)."""
    np, pd = _pd()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import blended_transform_factory
    an = run_root / "analysis/revision2"
    data = folder / "source_data"
    data.mkdir(parents=True, exist_ok=True)
    rows_spec = [(m, (bar_labels or {}).get(m, label), group) for m, label, group in ABLATION_ROWS]
    ablation = [m for m, _, _ in rows_spec]
    slope = [m for m, _ in SLOPE_ROWS]
    full = load_eval(run_root, list(dict.fromkeys(ablation + slope)))
    s = b.site_scores(full)
    s.to_csv(data / f"{prefix}_greenhouse_scores.csv", index=False)
    table = s.pivot(index="greenhouse", columns="model", values="NMAE").sort_index()
    checks = []

    # (A) values: means and CIs recomputed from results.csv, paired increases from E2's table.
    summary = pd.read_csv(an / "bitfi_variants_summary.csv").set_index("model")
    paired = pd.read_csv(an / "bitfi_variants_paired.csv").set_index("model")
    xs = _ablation_positions(rows_spec)
    rows = []
    for x, (m, label, group) in zip(xs, rows_spec):
        mu, lo, hi = ps.interval(table[m].to_numpy())
        row = dict(model=m, label=_one_line(label), group=group, position=x, NMAE=mu, CI_low=lo, CI_high=hi,
                   increase=0., increase_low=np.nan, increase_high=np.nan, holm_p=np.nan, significant=np.nan,
                   greenhouses_worse=np.nan)
        if bar_labels is not None:
            row["table_name"] = TABLE_VARIANT_NAMES[m]
        if m != MAIN_MODEL:
            q = paired.loc[m]
            red, rlo, rhi = paired_reduction(table[m], table[MAIN_MODEL])
            row.update(increase=q.change_pct, increase_low=q.change_low, increase_high=q.change_high, holm_p=q.holm_p,
                       significant=bool(q.holm_p < .05), greenhouses_worse=int(q.greenhouses_variant_worse),
                       recomputed_increase=-red, recomputed_low=-rhi, recomputed_high=-rlo)
        rows.append(row)
    a = pd.DataFrame(rows)
    checks.append(_agree("A top, bitfi_variants_summary.csv mean/low/high", a[["NMAE", "CI_low", "CI_high"]],
                         summary.loc[a.model, ["mean", "low", "high"]]))
    v = a[a.model != MAIN_MODEL]
    checks.append(_agree("A bottom, bitfi_variants_paired.csv change_pct/low/high vs recomputed paired bootstrap",
                         v[["recomputed_increase", "recomputed_low", "recomputed_high"]],
                         v[["increase", "increase_low", "increase_high"]], tol=1e-9))
    worse = [int((table[m] > table[MAIN_MODEL]).sum()) for m in v.model]
    checks.append(_agree("A bottom, greenhouses with higher NMAE than full BiTFI", worse, v.greenhouses_worse))

    # (B) values: greenhouse scores and the two paired reductions.
    gh = table[slope].copy()
    stored = pd.read_csv(run_root / "analysis/all_site_scores.csv").pivot(index="greenhouse", columns="model", values="NMAE")
    checks.append(_agree("B, all_site_scores.csv (TimesFM3 local + cross, BiTFI)", gh[["TimesFM3.0-COV-SPA", MAIN_MODEL]],
                         stored.loc[gh.index, ["TimesFM3.0-COV-SPA", MAIN_MODEL]]))
    stored = pd.read_csv(an / "covreg_sites.csv").set_index("greenhouse")
    checks.append(_agree("B, covreg_sites.csv (LightGBM local + cross, BiTFI)", gh[["CovReg-LGBM", MAIN_MODEL]],
                         stored.loc[gh.index, ["CovReg-LGBM", MAIN_MODEL]]))
    steps = []
    for (m, ref), name in [(("CovReg-LGBM", "TimesFM3.0-COV-SPA"), "covreg_paired_context.csv"),
                           ((MAIN_MODEL, "CovReg-LGBM"), "covreg_paired.csv")]:
        q = pd.read_csv(an / name).set_index(["model", "reference"]).loc[(m, ref)]
        red, lo, hi = paired_reduction(gh[m], gh[ref])
        steps.append(dict(model=m, reference=ref, source=name, reduction=q.reduction, low=q.low, high=q.high, p=q.p,
                          holm_p=q.holm_p, greenhouses_lower=int((gh[m] < gh[ref]).sum()), n_greenhouses=len(gh),
                          recomputed=red, recomputed_low=lo, recomputed_high=hi))
    st = pd.DataFrame(steps)
    checks.append(_agree("B, reductions in covreg_paired_context.csv and covreg_paired.csv vs recomputed",
                         st[["recomputed", "recomputed_low", "recomputed_high"]], st[["reduction", "low", "high"]], tol=1e-9))
    ids = test_ids(run_root)
    flagged = gh.index[gh[MAIN_MODEL] > gh["CovReg-LGBM"]].tolist()
    if len(flagged) != 1:
        raise ValueError(f"Expected one greenhouse where the LightGBM regression beats BiTFI, got {flagged}")
    flagged = flagged[0]

    with plt.rc_context(MATH_RC):
        W, H = 7.2, 4.25
        fig = plt.figure(figsize=(W, H))
        top, h_top, gap, h_bottom = H - .45, 1.95, .10, 1.30
        bottom = top - h_top - gap - h_bottom
        ax_t = fig.add_axes([.62 / W, (top - h_top) / H, 3.40 / W, h_top / H])
        ax_d = fig.add_axes([.62 / W, bottom / H, 3.40 / W, h_bottom / H])
        ax_b = fig.add_axes([4.70 / W, bottom / H, 2.18 / W, (top - bottom) / H])
        width = .78
        y_top = a.CI_high.max() * 1.17
        y_down = max(float(v.increase_high.max()) * 1.2, 10.)
        labels = []
        for r, (_, label, _) in zip(a.itertuples(), rows_spec):
            c = ps.COLORS[r.model]
            ax_t.bar(r.position, r.NMAE, width=width, color=c, lw=0, zorder=2)
            ax_t.errorbar(r.position, r.NMAE, yerr=[[r.NMAE - r.CI_low], [r.CI_high - r.NMAE]], fmt="none", ecolor=INK,
                          elinewidth=.9, capsize=2.5, capthick=.9, zorder=3)
            ax_t.annotate(f"{r.NMAE:.4f}", (r.position, r.CI_high), xytext=(0, 2.5), textcoords="offset points",
                          ha="center", va="bottom", fontsize=7, color=INK)
            labels.append((ax_t.annotate(label, (r.position, 0), xytext=(0, 4), textcoords="offset points", rotation=90,
                                         ha="center", va="bottom", multialignment="left", linespacing=1.2,
                                         fontsize=7.2, color="white", zorder=4), r.position, r.NMAE, r.CI_low))
            if r.model == MAIN_MODEL:
                # "Comparator", not "Reference", which the paper reserves for cross-greenhouse reference series.
                ax_d.annotate("Comparator", (r.position, 0), xytext=(0, -4), textcoords="offset points", rotation=90,
                              ha="center", va="top", fontsize=7, color=MUTED_INK)
                continue
            ax_d.bar(r.position, r.increase, width=width, color=c, alpha=.42, lw=0, zorder=2)
            ax_d.errorbar(r.position, r.increase, yerr=[[r.increase - r.increase_low], [r.increase_high - r.increase]],
                          fmt="none", ecolor=INK, elinewidth=.9, capsize=2.5, capthick=.9, zorder=3)
            text = _signed(r.increase) if r.significant else _signed(r.increase) + "\nn.s."
            ax_d.annotate(text, (r.position, r.increase_high), xytext=(0, -2.5), textcoords="offset points", ha="center",
                          va="top", fontsize=7, color=INK if r.significant else MUTED_INK, linespacing=1.25)
        ax_t.axhline(0, color="black", lw=1.05, zorder=3)
        ax_d.axhline(0, color="black", lw=1.05, zorder=3)
        ax_t.axhline(a.NMAE.iloc[0], color=ps.COLORS[MAIN_MODEL], lw=.8, ls=(0, (3, 2)), zorder=1)
        for ax in (ax_t, ax_d):
            ax.set_xlim(xs[0] - .6, xs[-1] + .6)
            ax.set_xticks([])
            _outward_left(ax)
        ax_t.set_ylim(0, y_top)
        ax_t.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax_d.set_ylim(y_down, -2.5)
        ax_d.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax_t.set_ylabel("NMAE")
        ax_d.set_ylabel("NMAE increase (%)")
        for ax in (ax_t, ax_d):
            ax.yaxis.set_label_coords(-.115, .5)
        # Group brackets under the mirrored bars; their labels share one baseline row with the
        # x tick labels of B (both ``drop`` points below the common bottom edge of the axes).
        blend = blended_transform_factory(ax_d.transData, ax_d.transAxes)
        drop = 9.
        per_pt = 1 / (h_bottom * 72)
        for key, name in ABLATION_GROUPS:
            members = a[a.group == key].position
            x0, x1 = members.min() - width / 2, members.max() + width / 2
            y0, y1 = -3 * per_pt, -5.5 * per_pt
            ax_d.plot([x0, x0, x1, x1], [y0, y1, y1, y0], transform=blend, color=MUTED_INK, lw=.8, clip_on=False)
            ax_d.text((x0 + x1) / 2, -drop * per_pt, name, transform=blend, ha="center", va="top", fontsize=7.2,
                      color=INK)
        panel_label(ps, ax_t, "A", "Information sources and references")

        # (B) slope graph
        x3 = np.arange(len(SLOPE_ROWS))
        for g, values in gh.iterrows():
            if g != flagged or not HIGHLIGHT_EXCEPTION:
                ax_b.plot(x3, values.to_numpy(), color=GREENHOUSE_LINE, lw=.9, marker="o", ms=2.6, mfc=GREENHOUSE_LINE,
                          mec="none", zorder=2)
        if HIGHLIGHT_EXCEPTION:
            ax_b.plot(x3, gh.loc[flagged].to_numpy(), color=HIGHLIGHT, lw=1.5, marker="o", ms=3.4, mec="white", mew=.5,
                      zorder=3)
        means = gh.mean()
        ax_b.plot(x3, means.to_numpy(), color="#2E3338", lw=2.4, zorder=4, solid_capstyle="round")
        for j, (m, _) in enumerate(SLOPE_ROWS):
            ax_b.plot(j, means[m], marker="o", ms=7.5, mfc=ps.COLORS[m], mec="white", mew=1.1, zorder=5)
        end = x3[-1]
        ax_b.annotate("Mean", (end, means[MAIN_MODEL]), xytext=(8, 0), textcoords="offset points", ha="left",
                      va="center", fontsize=7.2, color=INK, fontweight="bold")
        if HIGHLIGHT_EXCEPTION:
            ax_b.annotate(ids[flagged], (end, gh.loc[flagged, MAIN_MODEL]), xytext=(6, 0), textcoords="offset points",
                          ha="left", va="center", fontsize=7.2, color=INK)
        y_lo = min(.02, float(gh.min().min()) - .002)
        y_hi = float(gh.max().max())
        ax_b.set_ylim(y_lo, y_hi + (y_hi - y_lo) * .2)
        ax_b.set_xlim(-.32, end + .62)
        for j, r in enumerate(st.itertuples()):
            xm = j + .5
            yb = y_hi + (y_hi - y_lo) * .065
            ax_b.plot([j + .1, j + .1, j + .9, j + .9], [yb - .0006, yb, yb, yb - .0006], color=MUTED_INK, lw=.8)
            ax_b.annotate(_signed(-r.reduction, 2), (xm, yb), xytext=(0, 9.5), textcoords="offset points",
                          ha="center", va="bottom", fontsize=7.6, color=INK, fontweight="bold")
            ax_b.annotate(f"{r.greenhouses_lower} of {r.n_greenhouses} lower", (xm, yb), xytext=(0, 2), textcoords="offset points",
                          ha="center", va="bottom", fontsize=6.6, color=MUTED_INK)
        ax_b.set_xticks(x3, [label for _, label in SLOPE_ROWS])
        ax_b.tick_params(axis="x", length=0, pad=drop, labelsize=7.2)
        ax_b.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
        ax_b.set_ylabel("Greenhouse NMAE")
        _outward_left(ax_b)
        panel_label(ps, ax_b, "B", "Equal-information control")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fits = []
        for text, x, height, low in labels:
            box = text.get_window_extent(renderer)
            top_px = ax_t.transData.transform((x, height))[1]
            # Lower CI cap (capthick .9 pt) drawn across the label column; the label must end below it.
            cap_px = ax_t.transData.transform((x, low))[1] - .45 * fig.dpi / 72
            fits.append(dict(position=x, label_top_px=box.y1, bar_top_px=top_px, margin_px=top_px - box.y1,
                             cap_clearance_px=cap_px - box.y1,
                             label_width_px=box.width, bar_width_px=ax_t.transData.transform((x + width / 2, 0))[0]
                             - ax_t.transData.transform((x - width / 2, 0))[0]))
        fits = pd.DataFrame(fits)
        if (fits.margin_px < 2).any() or (fits.label_width_px > fits.bar_width_px - 2).any():
            raise ValueError(f"In-bar labels do not fit:\n{fits}")
        if (fits.cap_clearance_px < 0).any():
            raise ValueError(f"In-bar labels run into the lower CI caps:\n{fits}")
        checks.append(f"layout: all in-bar labels inside their bars (minimum margin {fits.margin_px.min():.1f} px "
                      f"above, {float((fits.bar_width_px - fits.label_width_px).min()):.1f} px across at {fig.dpi:.0f} dpi)")
        checks.append(f"layout: in-bar labels end below the lower CI caps (minimum clearance "
                      f"{fits.cap_clearance_px.min():.1f} px at {fig.dpi:.0f} dpi)")

    a.to_csv(data / f"{prefix}_A_ablation.csv", index=False)
    gh.assign(test_id=[ids[g] for g in gh.index], highlighted=[HIGHLIGHT_EXCEPTION and g == flagged for g in gh.index]).to_csv(
        data / f"{prefix}_B_greenhouse_nmae.csv")
    st.to_csv(data / f"{prefix}_B_paired_reductions.csv", index=False)
    (data / f"{prefix}_crosscheck.txt").write_text("\n".join(checks) + "\n")
    rec.save(fig, stem, folder=folder, png=True)


def s15r_inputs(run_root):
    an = run_root / "analysis"
    return [run_root / "evaluation" / ALIASES.get(m, m) / "results.csv" for m in CONTROL_MODELS] + [
        an / "revision2/covreg_scenario_gap.csv", an / "all_gaps.csv"]


def build_s15_redesign(ps, b, run_root, folder, rec, stem="FigureS14_Equal_information_by_gap_preview",
                       prefix="FigureS15r"):
    """Fig. 7C-E of the first revision-2 draft as a supplementary figure, drawn by the same
    ``gap_lines`` code (equal-information controls by gap length, one panel per scenario).
    ``stem`` and ``prefix`` name the figure and its source tables."""
    np, pd = _pd()
    import matplotlib.pyplot as plt
    data = folder / "source_data"
    data.mkdir(parents=True, exist_ok=True)
    controls = list(CONTROL_MODELS)
    full = load_eval(run_root, controls)
    g = b.site_scores(full, ["scenario", "gap_length_h"])
    g.to_csv(data / f"{prefix}_gap_scenario_greenhouse_scores.csv", index=False)
    fig = plt.figure(figsize=(7.2, 3.3))
    axes = [fig.add_axes([.085 + j * .305, .15, .265, .60]) for j in range(3)]
    lines = []
    for j, sc in enumerate(["A", "B", "C"]):
        lines.append(gap_lines(ps, axes[j], g[g.scenario == sc], controls, CONTROL_LABELS).assign(scenario=sc))
        panel_label(ps, axes[j], chr(65 + j), SCENARIO_TITLES[sc])
    top_y = max(ax.get_ylim()[1] for ax in axes)
    for j, ax in enumerate(axes):
        ax.set_ylim(0, top_y)
        if j:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("NMAE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="center", bbox_to_anchor=(.53, .905), fontsize=7.5, handlelength=1.7,
               columnspacing=1.2)
    out = pd.concat(lines, ignore_index=True)
    an = run_root / "analysis"
    ref = pd.concat([pd.read_csv(an / "revision2/covreg_scenario_gap.csv"),
                     pd.read_csv(an / "all_gaps.csv").query("model == 'TimesFM3.0-COV-SPA'")], ignore_index=True)
    m = out.merge(ref, on=["model", "scenario", "gap_length_h"], suffixes=("", "_table"), validate="one_to_one")
    if len(m) != len(out):
        raise ValueError("FigureS15r: unmatched model-scenario-gap cells")
    checks = [_agree("covreg_scenario_gap.csv and all_gaps.csv mean/low/high", m[["mean", "CI_low", "CI_high"]],
                     m[["mean_table", "low", "high"]])]
    out.to_csv(data / f"{prefix}_gap_summary.csv", index=False)
    (data / f"{prefix}_crosscheck.txt").write_text("\n".join(checks) + "\n")
    rec.save(fig, stem, folder=folder / "Supplementary", png=True)


def fig7r_inputs(run_root):
    an = run_root / "analysis/revision2"
    files = [an / f"indicators_{k}_{t}{c}.csv" for k in ["vpd", "radiation"] for t in ["days", "summary"] for c in ["", "_covreg"]]
    files += [an / f"indicators_vpd_cases{c}.csv" for c in ["", "_covreg"]]
    return files + [run_root / "mask_manifest.csv"] + [run_root / "evaluation" / m / "results.csv" for m, _ in SCATTER_ROWS]


def hourly_vpd_points(run_root, models):
    """Hourly observed and reconstructed VPD of the gaps that mask Tin and RH together (Scenarios B and C).

    Same inputs and formulas as E5 (``revision2_indicators``: stored predictions, float64 inverse scaling,
    Tetens VPD). Also returns the daily means of complete calendar days for the cross-check with E5's
    day table."""
    np, pd = _pd()
    import clean_protocol as cp
    import revision2_indicators as rv
    if rv.RUN != run_root.resolve():
        raise SystemExit(f"revision2_indicators is bound to {rv.RUN}, not {run_root}")
    manifest = pd.read_csv(run_root / "mask_manifest.csv")
    cases = manifest[manifest.masked_vars.str.split(",").map(lambda v: {"Tin", "RH"} <= set(v))]
    sites = {name: cp.site(name) for name in cases.greenhouse.unique()}
    hourly, daily = [], []
    for m in models:
        read, *_ = rv.prediction_reader(m, manifest)
        for r in cases.itertuples():
            site = sites[r.greenhouse]
            interval = slice(int(r.start_idx), int(r.end_idx) + 1)
            p = np.asarray(read(r))
            t = rv.physical(site["data"]["Tin"].iloc[interval].to_numpy(), site["scaler"]["Tin"])
            h = rv.physical(site["data"]["RH"].iloc[interval].to_numpy(), site["scaler"]["RH"])
            te = rv.physical(p[:, cp.COLS.index("Tin")], site["scaler"]["Tin"])
            he = rv.physical(p[:, cp.COLS.index("RH")], site["scaler"]["RH"])
            obs, est = rv.vapour_pressure_deficit(t, h), rv.vapour_pressure_deficit(te, he)
            hourly.append(pd.DataFrame(dict(model=m, greenhouse=r.greenhouse, case_id=int(r.case_id), scenario=r.scenario,
                                            observed=obs, reconstructed=est)))
            for day, positions in rv.complete_days(site["data"].index[interval]):
                daily.append(dict(model=m, case_id=int(r.case_id), date=day.strftime("%Y-%m-%d"),
                                  observed=float(obs[positions].mean()), reconstructed=float(est[positions].mean())))
    return pd.concat(hourly, ignore_index=True), pd.DataFrame(daily)


def build_fig7_redesign(ps, b, run_root, folder, rec, stem="Figure7_Agricultural_indicators_preview",
                        prefix="Figure7r"):
    """New Fig. 7 candidate in the form of Moon Fig. 4: observed versus reconstructed density
    (hexagon counts, logarithmic shading) for hourly VPD, daily mean VPD and daily radiation sums,
    one row per configuration, with the 1:1 line and the least-squares line. ``stem`` and
    ``prefix`` name the figure and its source tables."""
    np, pd = _pd()
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.ticker import FuncFormatter
    an = run_root / "analysis/revision2"
    data = folder / "source_data"
    data.mkdir(parents=True, exist_ok=True)
    models = [m for m, _ in SCATTER_ROWS]
    checks = []

    def e5(indicator, kind):
        main = pd.read_csv(an / f"indicators_{indicator}_{kind}.csv")
        extra = pd.read_csv(an / f"indicators_{indicator}_{kind}_covreg.csv")
        return pd.concat([main, extra[extra.model.str.startswith("CovReg-")]], ignore_index=True)

    hourly, recomputed_days = hourly_vpd_points(run_root, models)
    cases = e5("vpd", "cases").query("variant == 'primary'")
    case_mae = hourly.assign(ae=(hourly.reconstructed - hourly.observed).abs()).groupby(["model", "case_id"]).ae.mean()
    stored = cases.set_index(["model", "case_id"]).VPD_MAE.loc[case_mae.index]
    checks.append(_agree("hourly VPD case MAE vs indicators_vpd_cases(_covreg).csv", case_mae, stored, tol=1e-12))
    vpd_days = e5("vpd", "days").query("variant == 'primary'")
    merged = recomputed_days.merge(vpd_days, on=["model", "case_id", "date"], validate="one_to_one")
    if len(merged) != len(recomputed_days) or len(merged) != len(vpd_days[vpd_days.model.isin(models)]):
        raise ValueError("Daily VPD: recomputed complete days do not match indicators_vpd_days")
    checks.append(_agree("daily VPD from the hourly points vs indicators_vpd_days(_covreg).csv",
                         merged[["observed", "reconstructed"]], merged[["VPD_daily_observed", "VPD_daily_reconstructed"]],
                         tol=1e-12))
    rad_days = e5("radiation", "days").query("variant == 'primary'")
    sources = {
        "hourly_vpd": hourly,
        "daily_vpd": vpd_days.rename(columns={"VPD_daily_observed": "observed", "VPD_daily_reconstructed": "reconstructed"}),
        "daily_radiation": rad_days.rename(columns={"H_observed": "observed", "H_reconstructed": "reconstructed"}),
    }
    summaries = {k: e5(k, "summary").query("variant == 'primary'") for k in ["vpd", "radiation"]}

    metrics, kept = [], []
    for key, indicator, metric, scope, scenarios, *_ in SCATTER_COLUMNS:
        d = sources[key]
        d = d[d.model.isin(models) & d.scenario.isin(scenarios)]
        first = d[d.model == models[0]].observed.to_numpy()
        table = summaries[indicator].query("metric == @metric and scope == @scope").set_index("model")
        for m in models:
            z = d[d.model == m]
            if not np.array_equal(z.observed.to_numpy(), first):
                raise ValueError(f"{key}: observed values differ between configurations ({m})")
            error = z.reconstructed - z.observed
            site_mae = error.abs().groupby(z.greenhouse).mean().mean()
            residual = float((error ** 2).sum())
            total = float(((z.observed - z.observed.mean()) ** 2).sum())
            slope, intercept = np.polyfit(z.observed, z.reconstructed, 1)
            metrics.append(dict(column=key, model=m, metric=metric, scope=scope, n_points=len(z),
                                n_greenhouses=z.greenhouse.nunique(), MAE_E5=float(table.loc[m, "mean"]),
                                MAE_E5_low=float(table.loc[m, "low"]), MAE_E5_high=float(table.loc[m, "high"]),
                                MAE_recomputed=float(site_mae), MAE_pooled=float(error.abs().mean()),
                                R2=1 - residual / total, slope=float(slope), intercept=float(intercept),
                                bias=float(error.mean())))
            kept.append(z[["model", "greenhouse", "case_id", "scenario", "observed", "reconstructed"]].assign(column=key))
    metrics = pd.DataFrame(metrics)
    checks.append(_agree("panel MAE (mean of greenhouse MAEs) vs indicators_*_summary(_covreg).csv",
                         metrics.MAE_recomputed, metrics.MAE_E5, tol=1e-12))

    with plt.rc_context(MATH_RC):
        W = 7.2
        left, panel, gap_x, gap_y, head, foot = 1.45, 1.41, .42, .14, .44, .80
        H = head + 4 * panel + 3 * gap_y + foot
        fig = plt.figure(figsize=(W, H))
        axes = [[fig.add_axes([(left + j * (panel + gap_x)) / W, (foot + (3 - i) * (panel + gap_y)) / H, panel / W,
                               panel / H]) for j in range(3)] for i in range(4)]
        halo = [pe.Stroke(linewidth=2.3, foreground="white"), pe.Normal()]
        for j, (key, _, _, _, _, title, unit, lims, ticks, across, fmt, ramp) in enumerate(SCATTER_COLUMNS):
            cmap = LinearSegmentedColormap.from_list(ramp, RAMPS[ramp])
            extent = (lims[0], lims[1], lims[0], lims[1])
            collections = []
            for i, (m, row_label) in enumerate(SCATTER_ROWS):
                ax = axes[i][j]
                z = [k for k in kept if k.column.iloc[0] == key and k.model.iloc[0] == m][0]
                inside = z.observed.between(*lims) & z.reconstructed.between(*lims)
                q = metrics[(metrics.column == key) & (metrics.model == m)].index[0]
                metrics.loc[q, "n_outside_axes"] = int((~inside).sum())
                hb = ax.hexbin(z.observed[inside], z.reconstructed[inside], gridsize=across, extent=extent, cmap=cmap,
                               mincnt=1, linewidths=.15, edgecolors="face", zorder=1)
                collections.append(hb)
                r = metrics.loc[q]
                grid = np.array(lims, float)
                ax.plot(grid, grid, color="#2E3338", lw=.85, ls=(0, (3.5, 2.2)), zorder=3, path_effects=halo)
                ax.plot(grid, r.intercept + r.slope * grid, color="#1F2429", lw=.95, zorder=4, path_effects=halo)
                ax.text(.05, .95, f"MAE {fmt.format(r.MAE_E5)}\n$R^2$ {r.R2:.2f}", transform=ax.transAxes, ha="left",
                        va="top", fontsize=6.9, color=INK, linespacing=1.25, zorder=5,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
                ax.set_xlim(*lims)
                ax.set_ylim(*lims)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.tick_params(labelsize=7, length=2.8, pad=2)
                if i < 3:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(f"Observed ({unit})", fontsize=8.2, labelpad=3)
                if j == 0:
                    ax.set_ylabel(f"Reconstructed ({unit})", fontsize=8.2, labelpad=3)
                    ax.annotate(row_label, xy=(0, .5), xycoords="axes fraction", xytext=(-40, 0), textcoords="offset points",
                                ha="right", va="center", fontsize=8.5, fontweight="bold", color=INK, linespacing=1.15)
            vmax = max(float(c.get_array().max()) for c in collections)
            norm = LogNorm(vmin=1, vmax=vmax)
            for c in collections:
                c.set_norm(norm)
            metrics.loc[metrics.column == key, "max_count_per_hexagon"] = vmax
            panel_label(ps, axes[0][j], chr(65 + j), title)
            if j == 0:
                from matplotlib.lines import Line2D
                keys = [Line2D([0], [0], color="#2E3338", lw=.85, ls=(0, (3.5, 2.2)), label="1:1 line"),
                        Line2D([0], [0], color="#1F2429", lw=.95, label="Least-squares fit")]
                fig.legend(handles=keys, loc="upper right", bbox_to_anchor=((left - .62) / W, (foot - .40) / H),
                           fontsize=7, handlelength=2.2, labelspacing=.45, borderaxespad=0, frameon=False)
            ref = axes[3][j].get_position()
            cax = fig.add_axes([ref.x0 + ref.width * .1, (foot - .62) / H, ref.width * .8, .06 / H])
            bar = fig.colorbar(collections[0], cax=cax, orientation="horizontal")
            bar.outline.set_linewidth(.5)
            bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            bar.ax.tick_params(labelsize=6.6, length=2, pad=1.5, width=.6)
            bar.ax.tick_params(which="minor", length=1.2, width=.5)
            bar.ax.set_title("Hours per hexagon" if key == "hourly_vpd" else "Days per hexagon", fontsize=7,
                             fontweight="normal", pad=2.5, color=INK)
        recorded = metrics.copy()
    kept = pd.concat(kept, ignore_index=True)
    kept[kept.column == "hourly_vpd"].drop(columns="column").to_csv(data / f"{prefix}_hourly_VPD_points.csv.gz", index=False)
    kept[kept.column != "hourly_vpd"].to_csv(data / f"{prefix}_daily_points.csv", index=False)
    recorded.to_csv(data / f"{prefix}_panel_metrics.csv", index=False)
    (data / f"{prefix}_crosscheck.txt").write_text("\n".join(checks) + "\n")
    rec.save(fig, stem, folder=folder, png=True)


def s11h_inputs(run_root):
    an = run_root / "analysis"
    return detail_inputs(run_root) + [an / "revision2/bitfi_variants_gaps.csv", an / "revision2/bitfi_variants_scenarios.csv",
                                      an / "revision2/covreg_scenario_gap.csv", an / "revision2/covreg_scenario.csv",
                                      an / "all_gaps.csv", an / "all_scenarios.csv"]


def build_s11_heatmap(ps, b, run_root, folder, rec, stem="FigureS11_Controls_heatmap_preview", prefix="FigureS11h",
                      heat_labels=None, group=None):
    """Fig. S11 alternative: mean greenhouse NMAE of the ten control configurations for every
    scenario and gap length (and all gap lengths together), written in the cells over a
    continuous single-hue scale shared by the three panels.

    Final render options: ``stem``/``prefix`` name the figure and its source tables,
    ``heat_labels`` (configuration -> row label, line breaks included) replaces the preview row
    labels, and ``group`` = (text, first, last configuration) adds a bracket left of those rows."""
    np, pd = _pd()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    data = folder / "source_data"
    data.mkdir(parents=True, exist_ok=True)
    rows = keep_rows(run_root, DETAIL_ROWS)
    models = [m for m, _ in rows]
    full = load_eval(run_root, models)
    gap = b.site_scores(full, ["scenario", "gap_length_h"]).groupby(["model", "scenario", "gap_length_h"]).NMAE.mean()
    scen = b.site_scores(full, ["scenario"]).groupby(["model", "scenario"]).NMAE.mean()
    cells = []
    for m in models:
        for sc in ["A", "B", "C"]:
            for h in GAPS:
                cells.append(dict(model=m, scenario=sc, column=str(h), gap_length_h=h, NMAE=gap.loc[(m, sc, h)]))
            cells.append(dict(model=m, scenario=sc, column="All", gap_length_h=np.nan, NMAE=scen.loc[(m, sc)]))
    cells = pd.DataFrame(cells)
    an = run_root / "analysis"
    ref_gap = pd.concat([pd.read_csv(an / "revision2/bitfi_variants_gaps.csv"),
                         pd.read_csv(an / "revision2/covreg_scenario_gap.csv").query("model != @MAIN_MODEL"),
                         pd.read_csv(an / "all_gaps.csv").query("model == 'TimesFM3.0-COV-SPA'")], ignore_index=True)
    ref_sc = pd.concat([pd.read_csv(an / "revision2/bitfi_variants_scenarios.csv"),
                        pd.read_csv(an / "revision2/covreg_scenario.csv").query("model != @MAIN_MODEL"),
                        pd.read_csv(an / "all_scenarios.csv").query("model == 'TimesFM3.0-COV-SPA'")], ignore_index=True)
    g = cells.dropna(subset=["gap_length_h"]).merge(ref_gap, on=["model", "scenario", "gap_length_h"], validate="one_to_one")
    s = cells[cells.column == "All"].merge(ref_sc, on=["model", "scenario"], validate="one_to_one")
    if len(g) != 15 * len(models) or len(s) != 3 * len(models):
        raise ValueError("FigureS11 heatmap: cells missing from the analysis tables")
    checks = [_agree("scenario x gap cells vs bitfi_variants_gaps.csv, covreg_scenario_gap.csv, all_gaps.csv", g.NMAE, g["mean"]),
              _agree("scenario cells vs bitfi_variants_scenarios.csv, covreg_scenario.csv, all_scenarios.csv", s.NMAE, s["mean"])]
    cmap = LinearSegmentedColormap.from_list("nmae", RAMPS["nmae"])
    norm = Normalize(vmin=float(cells.NMAE.min()), vmax=float(cells.NMAE.max()))
    columns = [str(h) for h in GAPS] + ["All"]
    fig = plt.figure(figsize=(7.2, 3.9))
    left, gap_w = .14, .035
    width = (.99 - left - 2 * gap_w) / 3
    axes = [fig.add_axes([left + j * (width + gap_w), .22, width, .67]) for j in range(3)]
    # Two-line row labels (break before "(" or after the first comma, as the dot-plot labels of Fig. 3)
    # leave each cell about 22 pt for a 0.0000 value.
    wrapped = {"BiTFI-cross": "BiTFI,\ncalendar + cross", "BiTFI-random3": "BiTFI, 3 random\nreferences",
               "BiTFI-exclTop5": "BiTFI, 5 most-used\nexcluded"}
    row_labels = [wrapped.get(m, label.replace(" (", "\n(", 1)) for m, label in rows]
    if heat_labels is not None:
        row_labels = [heat_labels[m] for m in models]
    for j, sc in enumerate(["A", "B", "C"]):
        ax = axes[j]
        z = cells[cells.scenario == sc].pivot(index="model", columns="column", values="NMAE").loc[models, columns]
        image = ax.imshow(z.to_numpy(), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        best = z.to_numpy().argmin(axis=0)
        for i in range(len(models)):
            for k in range(len(columns)):
                value = z.iat[i, k]
                red, green, blue, _ = cmap(norm(value))
                light = .2126 * red ** 2.2 + .7152 * green ** 2.2 + .0722 * blue ** 2.2
                ax.text(k, i, f"{value:.4f}", ha="center", va="center", fontsize=6,
                        color="white" if light < .2 else INK)
        for k, i in enumerate(best):  # lowest NMAE of each column, outlined
            ax.add_patch(plt.Rectangle((k - .47, i - .43), .94, .86, fill=False, edgecolor=INK, lw=.9, zorder=3))
        ax.set_xticks(range(len(columns)), columns)
        ax.set_yticks(range(len(models)), row_labels if j == 0 else [])
        ax.tick_params(length=0, labelsize=7, pad=3)
        if j == 0:
            for t in ax.get_yticklabels():
                t.set_linespacing(1.2)
        for sep in [2.5, 5.5, 8.5]:
            ax.axhline(sep, color="white", lw=1.6)
        ax.axvline(4.5, color="white", lw=1.6)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("Gap length (h)", fontsize=8)
        panel_label(ps, ax, chr(65 + j), SCENARIO_TITLES[sc])
    if group is not None:
        group_bracket(axes[0], models.index(group[1]), models.index(group[2]), group[0])
    centre = axes[1].get_position()
    cax = fig.add_axes([centre.x0 + centre.width * .05, .045, centre.width * .9, .022])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=6.8, length=2, pad=1.5)
    bar.ax.set_title("Mean greenhouse NMAE", fontsize=7.2, fontweight="normal", pad=2.5, color=INK)
    if heat_labels is not None:
        cells["row_label"] = cells.model.map({m: _one_line(heat_labels[m]) for m in models})
    cells.to_csv(data / f"{prefix}_cells.csv", index=False)
    (data / f"{prefix}_crosscheck.txt").write_text("\n".join(checks) + "\n")
    rec.save(fig, stem, folder=folder / "Supplementary", png=True)


REDESIGN_FIGURES = {
    "fig6r": (build_fig6_redesign, fig6r_inputs),
    "fig7r": (build_fig7_redesign, fig7r_inputs),
    "s15r": (build_s15_redesign, s15r_inputs),
    "s11h": (build_s11_heatmap, s11h_inputs),
}


# ---------------------------------------------------------------------------
# Final manuscript figures (F3, 2026-10-20). The author adopted the redesign previews; the old
# Fig. 6 (reconstruction example) becomes Fig. 8. Same builders as the previews, with the variant
# names of the manuscript tables and the manuscript file names. ``final`` writes main figures to
# <out>, supplementary figures to <out>/Supplementary and source tables named after the figure
# to <out>/source_data, so they cannot be confused with the drafts in <out>/revision2 (old
# Figure7_Information_sources, Figure8_Agricultural_indicators and the dot-plot S11).
# ---------------------------------------------------------------------------

# Short names of the BiTFI variants in the manuscript tables (Table S16).
TABLE_VARIANT_NAMES = {MAIN_MODEL: "Top 3 (BiTFI)", "BiTFI-cross": "Calendar + cross", "BiTFI-local": "Local",
                       "BiTFI-calendar": "Calendar only", "BiTFI-k1": "Top 1", "BiTFI-random3": "Random 3",
                       "BiTFI-exclTop5": "Top 3, five most-used excluded"}
# Fig. 6A in-bar text. The information-source bars keep the Moon et al. (2021) form (what is
# removed) with the table name in parentheses. The longest name is broken at the hyphen of
# "most-used", which keeps it below the lower CI cap of its bar (checked in the builder).
FINAL_BAR_LABELS = {MAIN_MODEL: "Top 3 (BiTFI)", "BiTFI-cross": "−Local sensors\n(calendar + cross)",
                    "BiTFI-local": "−Cross references\n(local)", "BiTFI-calendar": "Calendar only",
                    "BiTFI-k1": "Top 1", "BiTFI-random3": "Random 3", "BiTFI-exclTop5": "Top 3, five most-\nused excluded"}
# Fig. S12 rows (one line each) and Fig. S11 heatmap rows (at most two lines).
FINAL_DETAIL_ROWS = [(m, TABLE_VARIANT_NAMES.get(m, label)) for m, label in DETAIL_ROWS]
FINAL_HEATMAP_LABELS = {m: label.replace(" (", "\n(", 1) for m, label in DETAIL_ROWS if m not in TABLE_VARIANT_NAMES}
FINAL_HEATMAP_LABELS.update(TABLE_VARIANT_NAMES, **{"BiTFI-exclTop5": FINAL_BAR_LABELS["BiTFI-exclTop5"]})
# Bracket over the six variants in S11 and S12, so that "Local" is not read as a forward
# TimesFM3 configuration; the full model is named "Top 3 (BiTFI)".
FINAL_GROUP = ("BiTFI variants", "BiTFI-calendar", "BiTFI-exclTop5")
# Fig. 8 is the author's PDF of the former Fig. 6 (2026-10-08), copied without any change.
RECONSTRUCTION_SOURCES = [ROOT / "05_thesis/Figure6_Reconstruction_example.pdf",
                          DEFAULT_RUN / "figures/Figure6_Reconstruction_example.pdf"]
RECONSTRUCTION_SHA256 = "dd2a51bca347f19b261c183fa8d04f193288c6df93bd3997a178cd74b077c8bf"


def group_bracket(ax, first, last, text, pad=5., tick=2.5, gap=2.5, inset=.12, fontsize=7.2):
    """Bracket and rotated group name left of the y tick labels of rows ``first`` to ``last`` of a
    categorical y axis. x is found from a draw and expressed in axes fraction, so the placement
    does not depend on the output resolution; the axes limits are checked to be unchanged."""
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory
    fig = ax.figure
    limits = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    lefts = [t.get_window_extent(renderer).x0 for loc, t in zip(ax.get_yticks(), ax.get_yticklabels())
             if first <= loc <= last and t.get_text()]
    if not lefts:
        raise ValueError("group_bracket: the bracketed rows have no tick labels")
    per_pt = fig.dpi / 72 / box.width  # axes fraction per point
    x = (min(lefts) - box.x0) / box.width - pad * per_pt
    blend = blended_transform_factory(ax.transAxes, ax.transData)
    y0, y1 = first - .5 + inset, last + .5 - inset
    ax.add_artist(Line2D([x + tick * per_pt, x, x, x + tick * per_pt], [y0, y0, y1, y1], transform=blend,
                         color=MUTED_INK, lw=.8, clip_on=False, solid_joinstyle="miter"))
    ax.text(x - gap * per_pt, (first + last) / 2, text, transform=blend, rotation=90, ha="right", va="center",
            fontsize=fontsize, color=INK, clip_on=False)
    if (tuple(ax.get_xlim()), tuple(ax.get_ylim())) != limits:
        raise ValueError("group_bracket changed the axes limits")


def reconstruction_inputs(run_root):
    return [next((p for p in RECONSTRUCTION_SOURCES if p.exists()), RECONSTRUCTION_SOURCES[0])]


def copy_reconstruction_example(ps, b, run_root, folder, rec):
    """Fig. 8: byte copy of the author's reconstruction example under its new number. The PDF is
    not redrawn. Its text layer is searched for figure numbers ("Fig. 6", "Figure 6"), which are
    reported in the render log, never edited."""
    import hashlib
    import re
    src = reconstruction_inputs(run_root)[0]
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    if digest != RECONSTRUCTION_SHA256:
        raise ValueError(f"{src} is not the author's PDF of 2026-10-08 (sha256 {digest})")
    dst = folder / "Figure8_Reconstruction_example.pdf"
    shutil.copy2(src, dst)
    if hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
        raise ValueError("Fig. 8 copy differs from its source")
    text = subprocess.run(["pdftotext", "-layout", str(dst), "-"], capture_output=True, text=True, check=True).stdout
    numbers = sorted(set(re.findall(r"(?i)\bfig(?:ure)?s?\.?\s*S?\d+[A-Za-z]?", text)))
    mentions = [line.strip() for line in text.splitlines() if re.search(r"(?i)fig", line)]
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(dst), str(folder / dst.stem)], check=True)
    rec.records[dst.stem] = dict(file=str(dst), source=str(src), sha256=digest, copied_unchanged=True,
                                 figure_numbers_in_text_layer=numbers, lines_mentioning_fig=mentions,
                                 words_in_text_layer=len(text.split()))
    rec.log_path.parent.mkdir(parents=True, exist_ok=True)
    rec.log_path.write_text(json.dumps(rec.records, indent=2, ensure_ascii=False))
    if numbers or mentions:
        print(f"[note] Fig. 8 text layer mentions a figure number: {numbers or mentions} (reported, not changed)",
              flush=True)


# ---------------------------------------------------------------------------
# Fig. S16 (2026-10-20): the largest BiTFI VPD error (case 2727, Test 06, Scenario C, 168 h).
# Inputs are the observed record and the directional components stored by
# ``revision2_failure_case.py`` (forward, backward and fused estimate of every pass).
# ---------------------------------------------------------------------------

FAILURE_CASE = 2727
FAILURE_FOLDER = "analysis/revision2/failure_case_2727"
FAILURE_ZOOM = ("2025-07-03 00:00", "2025-08-01 23:00")
# (components key, legend label, colour, line style, width); the final pass of each configuration.
FAILURE_LINES = [("BiTFI_fwd", "BiTFI, forward only", "#E2A0AE", "-", 1.3),
                 ("BiTFI_cross", "BiTFI, calendar + cross", "#71859D", "-", 1.3),
                 ("BiTFI", "BiTFI", "#C84E69", "-", 1.6)]
FAILURE_PASSES = [("BiTFI", "BiTFI", "#C84E69", "o", "-"), ("BiTFI_local", "Local", "#71859D", "^", "--"),
                  ("BiTFI_cross", "Calendar + cross", "#71859D", "s", "-"),
                  ("BiTFI_fwd", "Forward only", "#E2A0AE", "D", "-")]
FAILURE_STORED = {"BiTFI": MAIN_MODEL, "BiTFI_cross": "BiTFI-cross", "BiTFI_local": "BiTFI-local",
                  "BiTFI_fwd": "BiTFI-TimesFM3-fwd"}
VARIABLE_COLOURS = {"Tin": "#C99055", "Tout": "#76A9C6", "RH": "#6DA58D"}  # Fig. 1B
OBSERVED_GREY, TRUTH_INK, GAP_FILL, NATURAL_FILL = "#A3AAB1", "#1F2328", "#E4EEF7", "#F1F2F4"
FORWARD_COLOUR, BACKWARD_COLOUR = "#146F69", "#B8862B"


def failure_inputs(run_root):
    f = run_root / FAILURE_FOLDER
    return [f / "components.npz", f / "summary.csv", f / "complete.json", run_root / "mask_manifest.csv"] + [
        run_root / "evaluation" / m / "results.csv" for m in FAILURE_STORED.values()]


def build_failure_case(ps, b, run_root, folder, rec, stem="FigureS16_Failure_case", prefix="FigureS16_Failure_case"):
    """Observed record around the Test 06 gap, the final reconstructions of BiTFI, its forward-only arm and
    its calendar + cross variant, the pass-1 directional forecasts of indoor temperature and the hourly VPD
    error of every pass. Reconstructions are the stored components of the diagnostic rerun, which are
    checked against the stored evaluation predictions."""
    np, pd = _pd()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import clean_protocol as cp
    import revision2_indicators as rv
    data = folder / "source_data"
    data.mkdir(parents=True, exist_ok=True)
    f = run_root / FAILURE_FOLDER
    z = np.load(f / "components.npz")
    summary = pd.read_csv(f / "summary.csv")
    info = json.loads((f / "complete.json").read_text())
    row = pd.read_csv(run_root / "mask_manifest.csv").set_index("case_id").loc[FAILURE_CASE]
    site = cp.site(row.greenhouse)
    gs, ge = int(row.start_idx), int(row.end_idx)
    index = site["data"].index
    raw = site["data_raw"][cp.COLS]
    if str(index[gs]) != info["start_time"] or [str(t) for t in index[gs:ge + 1]] != list(z["time"]):
        raise ValueError("FigureS16: gap times differ between the manifest, the record and the components")
    gap_index = index[gs:ge + 1]
    col = {c: cp.COLS.index(c) for c in cp.COLS}
    phys = lambda a, c: rv.physical(a, site["scaler"][c])

    def fused(key, k=-1):
        return {c: phys(z[key][k, 2, col[c]], c) for c in cp.COLS}

    # Stored evaluation predictions agree with the final pass of each rerun configuration.
    checks = []
    for key, model in FAILURE_STORED.items():
        p = np.load(run_root / "evaluation" / model / "predictions.npz") if (
            run_root / "evaluation" / model / "predictions.npz").exists() else None
        stored = (p["prediction"][np.where(p["case_id"] == FAILURE_CASE)[0][0]] if p is not None else
                  np.load(run_root / "evaluation" / model / "predictions" / f"{FAILURE_CASE}.npy"))[:len(gap_index)]
        checks.append(_agree(f"{key} final pass vs stored {model} predictions (normalized)", z[key][-1, 2].T, stored, tol=1e-5))
    truth = {c: raw[c].iloc[gs:ge + 1].to_numpy(np.float64) for c in cp.COLS}
    vpd_truth = rv.vapour_pressure_deficit(truth["Tin"], truth["RH"])
    for key, _, _, _, _ in FAILURE_PASSES:
        for k in range(z[key].shape[0]):
            e = fused(key, k)
            mae = float(np.nanmean(np.abs(rv.vapour_pressure_deficit(e["Tin"], e["RH"]) - vpd_truth)))
            expected = summary.query("configuration == @key.replace('_', '-') and refinement_pass == @k").VPD_MAE
            checks.append(_agree(f"{key} pass {k} VPD MAE vs summary.csv", [mae], expected.to_numpy(), tol=1e-9))
    # Model input across the natural gap that follows: masked-input interpolation of the masked record.
    masked = site["data"][cp.COLS].copy()
    masked.iloc[gs:ge + 1] = np.nan
    bridge = masked.interpolate(limit_direction="both").ffill().bfill()
    after = np.arange(len(index)) > ge
    natural = raw.isna().all(axis=1).to_numpy() & after
    starts = np.flatnonzero(natural & ~np.r_[False, natural[:-1]])
    ends = np.flatnonzero(natural & ~np.r_[natural[1:], False])
    nat = [(index[s], index[e]) for s, e in zip(starts, ends) if e - s + 1 >= 24]
    if len(nat) != 1:
        raise ValueError(f"FigureS16: expected one natural gap of at least 24 h after the gap, found {nat}")
    nat0, nat1 = nat[0]
    hour = pd.Timedelta(hours=1)
    num = lambda t: mdates.date2num(pd.DatetimeIndex(t).to_pydatetime()) if not isinstance(t, pd.Timestamp) else \
        mdates.date2num(t.to_pydatetime())

    # Layout in inches from the top edge, so that titles, tick labels and legends keep fixed gaps.
    # The page holds the figure at about 0.83 of this size (7.9 in of A4 text height beside the caption).
    width_in, height_in = 7.2, 9.55
    fig = plt.figure(figsize=(width_in, height_in))
    left, width = .085, .885

    def box(x0, w, top_in, bottom_in):
        return [x0, 1 - bottom_in / height_in, w, (bottom_in - top_in) / height_in]
    ov = [fig.add_axes(box(left, width, t, t + .62)) for t in (.32, 1.34)]
    zoom_top = [2.78, 3.90, 5.02, 6.14]
    zm = [fig.add_axes(box(left, width, t, t + .72)) for t in zoom_top]
    for ax in zm[1:]:
        ax.sharex(zm[0])
    ax_dir = fig.add_axes(box(left, .52, 7.66, 8.96))
    ax_pass = fig.add_axes(box(.71, .26, 7.66, 8.96))

    def bands(ax, label=False):
        ax.axvspan(num(index[gs] - hour / 2), num(index[ge] + hour / 2), color=GAP_FILL, lw=0, zorder=0)
        for t in (index[gs] - hour / 2, index[ge] + hour / 2):
            ax.axvline(num(t), color="#7FA6C9", lw=.7, ls=":", zorder=1)
        ax.axvspan(num(nat0 - hour / 2), num(nat1 + hour / 2), facecolor=NATURAL_FILL, edgecolor="#C9CED3",
                   hatch="////", lw=0, zorder=0)

    # A, B: daily means of the whole record.
    daily = raw.resample("D").mean()
    dnum = num(daily.index + pd.Timedelta(hours=12))
    ax = ov[0]
    bands(ax)
    ax.plot(dnum, daily.Tout, color=VARIABLE_COLOURS["Tout"], lw=1.1, label="Outdoor temperature")
    ax.plot(dnum, daily.Tin, color=VARIABLE_COLOURS["Tin"], lw=1.3, label="Indoor temperature")
    below = (daily.Tin < daily.Tout).to_numpy()
    ax.fill_between(dnum, daily.Tin, daily.Tout, where=below, color=VARIABLE_COLOURS["Tout"], alpha=.18, lw=0,
                    interpolate=True, label="Indoor below outdoor")
    ax.set_ylabel("°C")
    panel_label(ps, ax, "A", "Temperature (daily means)")
    ax.legend(loc="lower right", bbox_to_anchor=(1., 1.02), ncol=3, fontsize=7.2, handlelength=1.6, columnspacing=1.2,
              borderaxespad=0)
    ax = ov[1]
    bands(ax)
    ax.plot(dnum, daily.RH, color=VARIABLE_COLOURS["RH"], lw=1.3)
    ax.set_ylabel("%")
    ax.set_ylim(30, 100)
    panel_label(ps, ax, "B", "Relative humidity (daily means)")
    for ax in ov:
        ax.set_xlim(dnum[0] - 2, dnum[-1] + 2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[9, 11, 1, 3, 5, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ov[0].tick_params(labelbottom=False)

    # C-F: hourly record around the gap with the final reconstructions.
    z0, z1 = pd.Timestamp(FAILURE_ZOOM[0]), pd.Timestamp(FAILURE_ZOOM[1])
    window = (index >= z0) & (index <= z1)
    wnum = num(index[window])
    gnum = num(gap_index)
    vpd_raw = rv.vapour_pressure_deficit(raw.Tin.to_numpy(np.float64), raw.RH.to_numpy(np.float64))
    rows = [("Tin", "C", "Indoor temperature", "°C"), ("Tout", "D", "Outdoor temperature", "°C"),
            ("RH", "E", "Relative humidity", "%"), ("VPD", "F", "Vapor pressure deficit", "kPa")]
    finals = {key: fused(key) for key, *_ in FAILURE_LINES}
    source = pd.DataFrame(dict(time=index[window].astype(str)))
    for ax, (var, letter, title, unit) in zip(zm, rows):
        bands(ax)
        if var == "VPD":
            obs = pd.Series(vpd_raw, index=index)
        else:
            obs = raw[var]
        context = obs.copy()
        context.iloc[gs:ge + 1] = np.nan
        ax.plot(wnum, context[window], color=OBSERVED_GREY, lw=.9, label="Observed")
        source[f"{var}_observed_outside_gap"] = context[window].to_numpy()
        if var != "VPD":
            fill = pd.Series(phys(bridge[var].to_numpy(), var), index=index)
            shown = fill.where(raw[var].isna() & after)
            ax.plot(wnum, shown[window], color=OBSERVED_GREY, lw=.9, ls=(0, (1, 1.6)), label="Interpolated model input")
            source[f"{var}_interpolated_input_after_gap"] = shown[window].to_numpy()
        t_obs = vpd_truth if var == "VPD" else truth[var]
        for key, label, colour, style, lw in FAILURE_LINES:
            e = finals[key]
            y = rv.vapour_pressure_deficit(e["Tin"], e["RH"]) if var == "VPD" else e[var]
            ax.plot(gnum, y, color=colour, ls=style, lw=lw, label=label, zorder=3)
            source[f"{var}_{key}"] = pd.Series(y, index=gap_index).reindex(index[window]).to_numpy()
        ax.plot(gnum, t_obs, color=TRUTH_INK, ls=(0, (3, 1.6)), lw=1.05, label="Masked observations", zorder=4)
        source[f"{var}_masked_observed"] = pd.Series(t_obs, index=gap_index).reindex(index[window]).to_numpy()
        ax.set_ylabel(unit)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        panel_label(ps, ax, letter, title)
    zm[0].set_xlim(num(z0), num(z1 + hour))
    zm[-1].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[3, 7, 11, 15, 19, 23, 27, 31]))
    zm[-1].xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    for ax in zm[:-1]:
        ax.tick_params(labelbottom=False)
    zm[-1].set_xlabel("Date (2025)")
    handles, labels = zm[0].get_legend_handles_labels()
    order = ["Observed", "Interpolated model input", "Masked observations", "BiTFI", "BiTFI, forward only",
             "BiTFI, calendar + cross"]
    pick = [handles[labels.index(n)] for n in order]
    pick += [Patch(facecolor=GAP_FILL, edgecolor="#7FA6C9", ls=":", lw=.7),
             Patch(facecolor=NATURAL_FILL, edgecolor="#C9CED3", hatch="////", lw=0)]
    fig.legend(pick, order + ["Evaluated gap (168 h)", "Natural gap (no record)"], ncol=4, loc="center",
               bbox_to_anchor=(.53, 1 - 2.38 / height_in), fontsize=7.2, handlelength=2.0, columnspacing=1.2)
    y1 = zm[0].get_ylim()[1]
    zm[0].text(num(nat0 + (nat1 - nat0) / 2), y1, "No record for 10 days", ha="center", va="top", fontsize=6.8,
               color=MUTED_INK, bbox=dict(facecolor=NATURAL_FILL, edgecolor="none", pad=1.2))

    # G: pass-1 directional forecasts of indoor temperature.
    ax = ax_dir
    ax.axvspan(gnum[0], gnum[-1], color=GAP_FILL, lw=0, zorder=0)
    k1 = 1
    fwd = phys(z["BiTFI"][k1, 0, col["Tin"]], "Tin")
    bwd = phys(z["BiTFI"][k1, 1, col["Tin"]], "Tin")
    fus = phys(z["BiTFI"][k1, 2, col["Tin"]], "Tin")
    bwd_cross = phys(z["BiTFI_cross"][k1, 1, col["Tin"]], "Tin")
    init = phys(z["BiTFI"][0, 2, col["Tin"]], "Tin")
    ax.plot(gnum, init, color=OBSERVED_GREY, lw=1.0, label="Initialization (pass 0)")
    ax.plot(gnum, fwd, color=FORWARD_COLOUR, lw=1.2, label="Forward forecast")
    ax.plot(gnum, bwd, color=BACKWARD_COLOUR, lw=1.2, label="Backward forecast")
    ax.plot(gnum, bwd_cross, color="#71859D", lw=1.1, ls=(0, (4, 1.5)), label="Backward, calendar + cross")
    ax.plot(gnum, fus, color="#C84E69", lw=1.6, label="Fused (BiTFI, pass 1)")
    ax.plot(gnum, truth["Tin"], color=TRUTH_INK, ls=(0, (3, 1.6)), lw=1.05, label="Masked observations")
    ax.set_xlim(gnum[0] - .1, gnum[-1] + .1)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    ax.set_ylabel("°C")
    ax.set_xlabel("Date (2025)")
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    panel_label(ps, ax, "G", "Indoor temperature forecasts, refinement pass 1")
    ax.legend(loc="upper left", fontsize=7, ncol=2, handlelength=1.8, columnspacing=1.0, borderaxespad=.2)
    ax.set_ylim(top=ax.get_ylim()[1] + .5 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
    pd.DataFrame(dict(time=gap_index.astype(str), initialization=init, forward_pass1=fwd, backward_pass1=bwd,
                      fused_pass1=fus, backward_pass1_calendar_cross=bwd_cross, masked_observed=truth["Tin"])
                 ).to_csv(data / f"{prefix}_pass1_indoor_temperature.csv", index=False)

    # H: hourly VPD MAE of every pass.
    ax = ax_pass
    passes = []
    for key, label, colour, marker, style in FAILURE_PASSES:
        name = key.replace("_", "-")
        s = summary[summary.configuration == name].sort_values("refinement_pass")
        ax.plot(s.refinement_pass, s.VPD_MAE, color=colour, marker=marker, ls=style, lw=1.2, ms=4,
                mfc="white" if key != "BiTFI" else colour, label=label)
        passes.append(s[["configuration", "refinement_pass", "VPD_MAE", "Tin_MAE", "RH_MAE"]])
    ax.set_xticks(range(int(summary.refinement_pass.max()) + 1))
    ax.set_xlabel("Refinement pass")
    ax.set_ylabel("VPD MAE (kPa)")
    ax.set_ylim(0, 1.)
    ax.set_yticks([0, .2, .4, .6])
    panel_label(ps, ax, "H", "Hourly VPD error by pass")
    ax.legend(loc="upper center", ncol=2, fontsize=6.8, handlelength=1.6, columnspacing=.7, borderaxespad=.1)
    pd.concat(passes).to_csv(data / f"{prefix}_vpd_by_pass.csv", index=False)
    source.to_csv(data / f"{prefix}_hourly_window.csv", index=False)
    daily.reset_index().to_csv(data / f"{prefix}_daily_record.csv", index=False)
    (data / f"{prefix}_crosscheck.txt").write_text("\n".join(checks) + "\n")
    rec.save(fig, stem, folder=folder / "Supplementary", png=True)


# Command keys keep the supplementary figure numbers used before the renumbering of 20 October 2026,
# whereas output file names carry the manuscript numbers. final: s11 -> Fig. S13, s12 -> Fig. S15,
# s15 -> Fig. S14, s16 -> Fig. S16. existing: s1 -> Fig. S2, s4 -> Fig. S11, s5 -> Fig. S3, s6 -> Fig. S12.
# new: s13 -> Fig. S9, s14 -> Fig. S8.
FINAL_FIGURES = {  # key: (builder, inputs, keyword arguments of the builder)
    "fig6": (build_fig6_redesign, fig6r_inputs, dict(stem="Figure6_Information_sources",
                                                     prefix="Figure6_Information_sources", bar_labels=FINAL_BAR_LABELS)),
    "fig7": (build_fig7_redesign, fig7r_inputs, dict(stem="Figure7_Agricultural_indicators",
                                                     prefix="Figure7_Agricultural_indicators")),
    "fig8": (copy_reconstruction_example, reconstruction_inputs, {}),
    "s11": (build_s11_heatmap, s11h_inputs, dict(stem="FigureS13_Controls_by_scenario",
                                                 prefix="FigureS13_Controls_by_scenario",
                                                 heat_labels=FINAL_HEATMAP_LABELS, group=FINAL_GROUP)),
    "s12": (build_detail_variable, detail_inputs, dict(rows=FINAL_DETAIL_ROWS, group=FINAL_GROUP)),
    "s15": (build_s15_redesign, s15r_inputs, dict(stem="FigureS14_Equal_information_by_gap",
                                                  prefix="FigureS14_Equal_information_by_gap")),
    "s16": (build_failure_case, failure_inputs, {}),
}


def render_final(args):
    run_root = Path(args.run_root).resolve()
    out = Path(args.out).resolve() if args.out else run_root / "figures"
    ps, b = configure(run_root, out, args.font)
    rec = Recorder(ps, None, out / "revision2/render_log_final.json", positions=True)
    ps.save = rec.save
    ps.setup()
    PARTIAL["on"] = False
    for name in args.only or list(FINAL_FIGURES):
        builder, needs, options = FINAL_FIGURES[name]
        missing = [str(p) for p in needs(run_root) if not p.exists()]
        if missing:
            raise SystemExit(f"{name}: missing inputs {missing}")
        builder(ps, b, run_root, out, rec, **options)
        print(f"[done] {name}", flush=True)


PARTIAL_OK = {"fig7", "fig7c", "s11", "s12"}
NEW_FIGURES = {
    "fig7": (build_fig7, fig7_inputs),
    "fig7c": (build_fig7_combined, fig7c_inputs),
    "fig8": (build_fig8, fig8_inputs),
    "s11": (build_detail_scenario, detail_inputs),
    "s12": (build_detail_variable, detail_inputs),
    "s13": (build_saits_seeds, saits_inputs),
    "s14": (build_loo, loo_inputs),
}


def render_new(args):
    run_root = Path(args.run_root).resolve()
    out = Path(args.out).resolve() if args.out else run_root / "figures"
    if args.stand_in:
        # Layout tests only: existing configurations stand in for unfinished ones.
        if _inside(out, ROOT / "03_result"):
            raise SystemExit("Stand-in renders must be written outside 03_result")
        ALIASES.update(dict(item.split("=", 1) for item in args.stand_in))
    ps, b = configure(run_root, out, args.font)
    folder = out / "revision2"
    rec = Recorder(ps, None, folder / "render_log_new.json")
    ps.save = rec.save
    ps.setup()
    chosen = args.only or list(NEW_FIGURES)
    PARTIAL["on"] = bool(args.allow_partial)
    redesign = None
    for name in chosen:
        if name in REDESIGN_FIGURES:
            # Redesign previews: own folder and render log; never part of the default run.
            if redesign is None:
                redesign = (out / REDESIGN_FOLDER, Recorder(ps, None, out / REDESIGN_FOLDER / "render_log_redesign.json"))
            builder, needs = REDESIGN_FIGURES[name]
            target, recorder = redesign
        else:
            builder, needs = NEW_FIGURES[name]
            target, recorder = folder, rec
        missing = [str(p) for p in needs(run_root) if not p.exists()]
        if missing and not (args.allow_partial and name in PARTIAL_OK):
            print(f"[skip] {name}: waiting for {missing}", flush=True)
            continue
        PARTIAL["dropped"] = []
        builder(ps, b, run_root, target, recorder)
        note = f" (provisional; rows without finished results: {PARTIAL['dropped']})" if PARTIAL["dropped"] else ""
        print(f"[done] {name}{note}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["existing", "new", "final"]:
        p = sub.add_parser(name)
        p.add_argument("--run-root", default=os.environ.get("BITFI_RUN_ROOT", str(DEFAULT_RUN)))
        p.add_argument("--out", default=None, help="Output folder (default <run root>/figures)")
        p.add_argument("--font", default=PUBLISHED_FONT,
                       help="Sans-serif family; default reproduces the published PDFs (Arial lacks U+2082, so CO₂ would lose its subscript)")
        if name == "existing":
            p.add_argument("--with-s6", action="store_true")
            p.add_argument("--published-s5b-title", action="store_true")
            p.add_argument("--no-copy-analysis", action="store_true")
            p.add_argument("--preview", default=None)
            p.add_argument("--log", default=None)
            p.add_argument("--order", choices=["published", "nmae"], default="published")
            p.add_argument("--only", nargs="*", default=None, help="Subset of fig3 fig4 fig5 s1 s4 s5")
            p.add_argument("--aligned-labels", action="store_true",
                           help="Panel letters on the title baseline (panel_label placement) instead of ps.panel")
        elif name == "new":
            p.add_argument("--only", nargs="*", default=None)
            p.add_argument("--stand-in", nargs="*", default=None, help="TARGET=EXISTING pairs for layout tests")
            p.add_argument("--allow-partial", action="store_true",
                           help="Draw fig7/s11/s12 without unfinished rows; files get a _provisional suffix")
        else:
            p.add_argument("--only", nargs="*", default=None, choices=list(FINAL_FIGURES),
                           help="Subset of the final manuscript figures (default all)")
    p = sub.add_parser("compare")
    for key in ["old", "new", "old_log", "new_log", "report"]:
        p.add_argument("--" + key.replace("_", "-"), required=True)
    args = parser.parse_args()
    {"existing": render_existing, "new": render_new, "final": render_final, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
