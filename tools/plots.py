#!/usr/bin/env python3
"""Turn results/*.csv into the report figures and tables.

Headline metric = KERNEL TIME = Atomic-Hook + Multi-Jump (hook_ms + jump_ms), i.e.
excluding the per-run worklist / light-list construction (setup_ms). Setup is always
shown separately (fig2) and every figure that uses the headline metric also has a
total-time counterpart in figures/total/, so conclusions can be checked both ways.

Outputs (results/figures/ unless --out is given):
  fig1_speedup.png        speedup over V0, all versions, all graphs (graphs ordered uniform -> skewed)
  fig2_breakdown.png      setup / Atomic-Hook / Multi-Jump time per version, per graph
  fig3_sweep_s.png        kernel time vs segment count s for V1, V3, V4 (dashed = auto s)
  fig4_counters.png       idle threads, SIMT lane efficiency, CAS retry rate per version
  fig5_skew.png           V4 speedup over V0 and over V1 vs degree skew (coefficient of variation)
  fig6_thresholds.png     V3 and V4 kernel time over the (t1, t2) grid, relative to each graph's best
  table_datasets.csv/.tex dataset table (from stats.csv)
  table_results.csv/.tex  median kernel time per graph x version (+ fastest variant)
  summary.csv             kernel, setup and total medians + speedups, every graph x version
  total/                  fig1, fig3, fig5 and table_results recomputed on total time
Usage: python3 tools/plots.py [--results results] [--out DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

VERS = ["V0", "V1", "V2", "V3", "V4"]
# Fixed-order categorical palette, validated for colour-vision deficiency and
# normal-vision separation. Colour follows the version, never its rank.
COL = {"V0": "#2a78d6", "V1": "#eb6834", "V2": "#1baf7a", "V3": "#eda100", "V4": "#6250d6"}
LABEL = {"V0": "V0 baseline", "V1": "V1 segmentation (Sutton-style)", "V2": "V2 degree-aware (ECL-CC-style)",
         "V3": "V3 combined", "V4": "V4 refined hybrid"}
PART = [("setup_ms", "#a9a8a2", "setup (worklists)"), ("hook_ms", "#2a78d6", "Atomic-Hook"),
        ("jump_ms", "#86b6ef", "Multi-Jump")]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # light -> dark
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "savefig.facecolor": "white"})


def style(ax, title=None, ylabel=None, xlabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_metrics(df):
    df = df.copy()
    df["kernel_ms"] = df.hook_ms + df.jump_ms
    return df


def graph_order(graphs, stats):
    """Uniform -> skewed (by degree coefficient of variation) when stats are available."""
    if stats is None:
        return sorted(graphs)
    cv = stats.set_index("graph").deg_cv
    return sorted(graphs, key=lambda g: cv.get(g, np.inf))


def grid(n, cols=4):
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    return rows, cols


# ------------------------------------------------------------------ tables
def write_tables(t, stats, out, metric):
    med = t.groupby(["graph", "version"])[metric].median().unstack()
    med = med[[v for v in VERS if v in med.columns]].reindex(graph_order(med.index, stats))
    res = med.round(2).copy()
    res["fastest"] = med.idxmin(axis=1)
    res.to_csv(os.path.join(out, "table_results.csv"))
    with open(os.path.join(out, "table_results.tex"), "w") as f:
        cols = list(med.columns)
        f.write("% median " + ("kernel time (Atomic-Hook + Multi-Jump)" if metric == "kernel_ms" else "total time") +
                " in ms; bold = fastest variant per graph\n")
        f.write("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\hline\nGraph & " + " & ".join(cols) + " \\\\\n\\hline\n")
        for g, r in med.iterrows():
            best = r.idxmin()
            cells = [(f"\\textbf{{{r[v]:.2f}}}" if v == best else f"{r[v]:.2f}") for v in cols]
            f.write(g.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    return med


def write_dataset_table(stats, out):
    if stats is None:
        return
    s = stats.set_index("graph").reindex(graph_order(stats.graph, stats))
    kind = {"road_usa": "road", "delaunay_n22": "mesh", "rgg_n_2_21_s0": "random geometric",
            "kron_g500-logn21": "synthetic power-law", "soc-LiveJournal1": "social", "com-Orkut": "social",
            "web-Google": "web", "europe_osm": "road", "indochina-2004": "web"}
    tab = pd.DataFrame({
        "type": [kind.get(g, "") for g in s.index],
        "vertices": s.n, "edges": s.m, "avg_degree": s.avg_deg.round(2), "max_degree": s.max_deg,
        "degree_cv": s.deg_cv.round(2), "components": s.components, "auto_s": s.auto_segments})
    tab.to_csv(os.path.join(out, "table_datasets.csv"))
    with open(os.path.join(out, "table_datasets.tex"), "w") as f:
        f.write("% |E| = undirected edges after symmetrising, deduplicating and removing self-loops;\n"
                "% degree CV = std/mean of degree; auto s = round(2|E|/|V|)\n")
        f.write("\\begin{tabular}{llrrrrrrr}\n\\hline\n"
                "Graph & Type & $|V|$ & $|E|$ & Avg deg & Max deg & Deg CV & Comps & $s$ \\\\\n\\hline\n")
        for g, r in tab.iterrows():
            f.write(f"{g.replace('_', chr(92) + '_')} & {r.type} & {r.vertices:,} & {r.edges:,} & {r.avg_degree:.2f} & "
                    f"{r.max_degree:,} & {r.degree_cv:.2f} & {r.components:,} & {r.auto_s} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")


def write_summary(t, out):
    md = t.groupby(["graph", "version"])[["kernel_ms", "setup_ms", "total_ms"]].median().unstack()
    s = pd.DataFrame(index=md.index)
    for v in VERS:
        if ("kernel_ms", v) in md:
            for m in ["kernel_ms", "setup_ms", "total_ms"]:
                s[f"{v}_{m.replace('_ms', '')}_ms"] = md[(m, v)]
    for metric in ["kernel", "total"]:
        for a, b in [("V0", "V1"), ("V0", "V2"), ("V0", "V3"), ("V1", "V3"), ("V2", "V3"),
                     ("V0", "V4"), ("V1", "V4"), ("V2", "V4"), ("V3", "V4")]:
            if f"{a}_{metric}_ms" in s and f"{b}_{metric}_ms" in s:
                s[f"{metric}_speedup_{b}_over_{a}"] = s[f"{a}_{metric}_ms"] / s[f"{b}_{metric}_ms"]
    s.round(4).to_csv(os.path.join(out, "summary.csv"))


# ------------------------------------------------------------------ fig 1
def fig_speedup(med, out, metric_label):
    graphs = list(med.index)
    vs = [v for v in VERS if v in med.columns]
    w = 0.82 / len(vs)
    x = np.arange(len(graphs))
    fig, ax = plt.subplots(figsize=(max(7, 1.25 * len(graphs) + 2), 4))
    top = 0
    for i, v in enumerate(vs):
        sp = (med["V0"] / med[v]).to_numpy()
        top = max(top, np.nanmax(sp))
        xs = x + (i - (len(vs) - 1) / 2) * w
        ax.bar(xs, sp, w * 0.9, color=COL[v], label=LABEL[v])
        if v == "V4":  # direct labels on the refined version only
            for xi, y in zip(xs, sp):
                ax.text(xi, y, f"{y:.1f}×", ha="center", va="bottom", fontsize=7, color=INK)
    ax.axhline(1.0, color=MUTED, linewidth=1)
    ax.set_ylim(0, top * 1.12)
    ax.set_xticks(x, graphs, rotation=20, ha="right")
    style(ax, f"Speedup over V0 — {metric_label} (higher is better)", "× faster than V0",
          "graphs ordered from uniform (left) to skewed degree distribution (right)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.32))
    save(fig, out, "fig1_speedup.png")


# ------------------------------------------------------------------ fig 2
def fig_breakdown(t, stats, out):
    md = t.groupby(["graph", "version"], as_index=False)[["setup_ms", "hook_ms", "jump_ms"]].median()
    graphs = graph_order(md.graph.unique(), stats)
    rows, cols = grid(len(graphs))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.7 * rows), squeeze=False)
    for ax, g in zip(axes.flat, graphs):
        d = md[md.graph == g].set_index("version").reindex([v for v in VERS if v in md.version.unique()])
        bottom = np.zeros(len(d))
        for p, c, lab in PART:
            vals = d[p].fillna(0).to_numpy()
            ax.bar(d.index, vals, 0.62, bottom=bottom, color=c, label=lab, edgecolor="white", linewidth=1.2)
            bottom += vals
        style(ax, g, "ms")
    for ax in axes.flat[len(graphs):]:
        ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Where the time goes: median time per phase", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, out, "fig2_breakdown.png")


# ------------------------------------------------------------------ fig 3
def fig_sweep(sw, stats, out, metric, metric_label):
    md = sw.groupby(["graph", "version", "segments"], as_index=False)[metric].median()
    graphs = graph_order(md.graph.unique(), stats)
    rows, cols = grid(len(graphs))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.7 * rows), squeeze=False)
    for ax, g in zip(axes.flat, graphs):
        for v in ["V1", "V3", "V4"]:
            d = md[(md.graph == g) & (md.version == v)].sort_values("segments")
            if len(d):
                ax.plot(d.segments, d[metric], color=COL[v], linewidth=2, marker="o", markersize=4,
                        markeredgecolor="white", label=LABEL[v])
        if stats is not None and g in set(stats.graph):
            ax.axvline(int(stats.set_index("graph").loc[g, "auto_segments"]), color=MUTED, linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        style(ax, g, "ms (log scale)", "segments s")
    for ax in axes.flat[len(graphs):]:
        ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{metric_label} vs number of segments (dashed line = automatic s = 2|E|/|V|)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, out, "fig3_sweep_s.png")


# ------------------------------------------------------------------ fig 4
def fig_counters(c, stats, out):
    c = c[c.simt_efficiency.notna() & c.version.isin(VERS)]
    if c.empty:
        return
    md = c.groupby(["graph", "version"], as_index=False)[
        ["simt_efficiency", "cas_retries", "cas_attempts", "idle_threads"]].median()
    graphs = graph_order(md.graph.unique(), stats)
    vs = [v for v in VERS if v in md.version.unique()]
    w = 0.82 / len(vs)
    x = np.arange(len(graphs))
    fig, axes = plt.subplots(3, 1, figsize=(max(8, 1.2 * len(graphs) + 2), 9), sharex=True)
    for i, v in enumerate(vs):
        d = md[md.version == v].set_index("graph").reindex(graphs)
        xs = x + (i - (len(vs) - 1) / 2) * w
        axes[0].bar(xs, d.idle_threads.clip(lower=1), w * 0.9, color=COL[v], label=LABEL[v])
        axes[1].bar(xs, d.simt_efficiency, w * 0.9, color=COL[v])
        axes[2].bar(xs, d.cas_retries / d.cas_attempts.clip(lower=1), w * 0.9, color=COL[v])
    axes[0].set_yscale("log")
    style(axes[0], "Idle threads launched (threads that processed no edge) — log scale", "threads")
    style(axes[1], "SIMT lane efficiency (useful lane-work / busy-warp lane slots; 1 = perfectly balanced)", "fraction")
    style(axes[2], "CAS retry rate (retries / attempts) — atomic contention", "fraction")
    axes[1].set_ylim(0, 1.05)
    axes[2].set_xticks(x, graphs, rotation=20, ha="right")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    save(fig, out, "fig4_counters.png")


# ------------------------------------------------------------------ fig 5
def fig_skew(med, stats, out, metric_label):
    if stats is None or "V4" not in med:
        return
    d = stats.set_index("graph").join(pd.DataFrame({
        "over_V0": med["V0"] / med["V4"], "over_V1": med["V1"] / med["V4"]}), how="inner")
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, v, lab in [("over_V0", "V0", "V4 speedup over V0 (baseline)"),
                        ("over_V1", "V1", "V4 speedup over V1 (segmentation only)")]:
        ax.scatter(d.deg_cv, d[col], s=60, color=COL[v], edgecolor="white", linewidth=1.5, zorder=3, label=lab)
    for g, r in d.iterrows():
        ax.annotate(g, (r.deg_cv, max(r.over_V0, r.over_V1)), fontsize=7, color=MUTED,
                    xytext=(5, 4), textcoords="offset points")
    ax.axhline(1.0, color=MUTED, linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}×"))
    style(ax, f"When does V4 pay off? Speedup vs degree skew — {metric_label}", "speedup (log scale)",
          "degree coefficient of variation = std / mean (log scale)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out, "fig5_skew.png")


# ------------------------------------------------------------------ fig 6
def fig_thresholds(st, stats, out, metric):
    md = st.groupby(["graph", "version", "t1", "t2"], as_index=False)[metric].median()
    md = md[md.version.isin(["V3", "V4"])]
    if md.empty:
        return
    graphs = graph_order(md.graph.unique(), stats)
    t1s = sorted(md.t1.unique())
    t2s = sorted(md.t2.unique())
    fig, axes = plt.subplots(2, len(graphs), figsize=(2.15 * len(graphs) + 1, 4.6), squeeze=False)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    vmax = 1.5
    im = None
    for j, g in enumerate(graphs):
        for i, v in enumerate(["V3", "V4"]):
            ax = axes[i, j]
            d = md[(md.graph == g) & (md.version == v)].pivot(index="t1", columns="t2", values=metric)
            d = d.reindex(index=t1s, columns=t2s)
            rel = d / np.nanmin(d.to_numpy())
            im = ax.imshow(rel.to_numpy(), cmap=cmap, vmin=1.0, vmax=vmax, aspect="auto")
            for (a, b), val in np.ndenumerate(rel.to_numpy()):
                if np.isfinite(val):
                    ax.text(b, a, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if val > 1.25 else INK)
            ax.set_xticks(range(len(t2s)), t2s, fontsize=7)
            ax.set_yticks(range(len(t1s)), t1s, fontsize=7)
            ax.tick_params(length=0, colors=MUTED)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(g, fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(f"{v}\nt1", color=MUTED)
            if i == 1:
                ax.set_xlabel("t2", color=MUTED, fontsize=8)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
    cb.set_label("time / best time for that graph & version", color=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("Sensitivity to dispatch thresholds (1.00 = best setting; darker = slower)",
                 x=0.01, ha="left", color=INK)
    save(fig, out, "fig6_thresholds.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "figures")
    out_total = os.path.join(out, "total")
    os.makedirs(out_total, exist_ok=True)
    p = lambda f: os.path.join(a.results, f)  # noqa: E731
    stats = pd.read_csv(p("stats.csv")) if os.path.exists(p("stats.csv")) else None
    write_dataset_table(stats, out)

    def load(name):
        if not os.path.exists(p(name)):
            return None
        d = pd.read_csv(p(name))
        d = d[d.version.isin(VERS)]
        if "correct" in d and (d.correct == 0).any():
            print(f"WARNING: {name}: {(d.correct == 0).sum()} runs had a WRONG partition; excluded")
            d = d[d.correct != 0]
        return add_metrics(d) if "hook_ms" in d else d

    t, sw, st, c = load("timing.csv"), load("sweep_s.csv"), load("sweep_t.csv"), load("counters.csv")
    if t is not None:
        write_summary(t, out)
        for metric, label, o in [("kernel_ms", "kernel time (Atomic-Hook + Multi-Jump)", out),
                                 ("total_ms", "total time (incl. setup)", out_total)]:
            med = write_tables(t, stats, o, metric)
            fig_speedup(med, o, label)
            fig_skew(med, stats, o, label)
        fig_breakdown(t, stats, out)
    if sw is not None:
        fig_sweep(sw, stats, out, "kernel_ms", "Kernel time (Atomic-Hook + Multi-Jump)")
        fig_sweep(sw, stats, out_total, "total_ms", "Total time (incl. setup)")
    if st is not None:
        fig_thresholds(st, stats, out, "kernel_ms")
    if c is not None:
        fig_counters(c, stats, out)
    print(f"figures and tables in {out} (total-time versions in {out_total})")


if __name__ == "__main__":
    main()
