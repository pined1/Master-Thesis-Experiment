#!/usr/bin/env python3
# ============================================================
# make_paper_figures.py
#
# Generates the three result figures used in the paper from the
# committed experiment results. All numbers below are the means
# reported in the paper's result tables, which are the values
# written by the experiment run.py scripts under experiments/*/results/:
#
#   Fig. 2  H1 sharing-scope ordering ....... exp01_h1_sharing_scope  (Table: h1-core)
#   Fig. 3  H1xH3 gate-vs-amplifier ......... h1xh3_crosssweep        (Table: h1xh3)
#   Fig. 4  H4 network-topology ranking ..... exp02_h4_topology       (Table: h4-ranking)
#
# Output: ./*.pdf  (copy the PDFs into the paper's figures/ folder).
#
# Run:  python figures/make_paper_figures.py
# ============================================================

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

# IEEE two-column friendly defaults (~3.4 in column width)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# Colorblind-safe palette (Wong)
# NONE=vermillion, LOCAL=orange, NEIGHBOR=blue, GLOBAL=green
C = {
    "NONE":     "#D55E00",
    "LOCAL":    "#E69F00",
    "NEIGHBOR": "#0072B2",
    "GLOBAL":   "#009E73",
    "neutral":  "#56B4E9",
    "hub":      "#D55E00",
}


# ---------------------------------------------------------------
# Fig. 2 — H1 sharing-scope ordering (Table h1-core, exp01)
# ---------------------------------------------------------------
def fig_h1_ordering():
    """Render Fig. 2: bar chart of mean incidents/org-year by sharing scope.

    Plots the four H1 sharing scopes (NONE/LOCAL/NEIGHBOR/GLOBAL) with a value
    label per bar and a 45.1% reduction annotation from NONE to GLOBAL. Saves
    fig_h1_ordering.pdf to OUT.
    """
    scopes = ["NONE", "LOCAL", "NEIGHBOR", "GLOBAL"]
    # mean incidents / org-year
    means  = [484.3, 406.4, 336.0, 265.6]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    bars = ax.bar(scopes, means, color=[C[s] for s in scopes], width=0.62)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 6, f"{m:.1f}",
                ha="center", va="bottom", fontsize=8)

    # 45.1% reduction annotation NONE -> GLOBAL
    ax.annotate("", xy=(3, 265.6), xytext=(0, 484.3),
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(2.05, 470, "45.1% fewer\nincidents", ha="center", va="center",
            fontsize=8, color="0.25")

    ax.set_ylabel("Mean incidents / org-year")
    ax.set_ylim(0, 540)
    ax.set_title("H1: Wider sharing scope lowers incidents")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_h1_ordering.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------
# Fig. 3 — H1xH3 gate-vs-amplifier (Table h1xh3, 50 seeds/cell)
# ---------------------------------------------------------------
def fig_h1xh3_curve():
    """Render Fig. 3: incidents vs prevention coefficient β, one line per scope.

    Shows the gate-vs-amplifier story: the NONE row stays flat (gate closed)
    while wider scopes drop steeply as β rises. Saves fig_h1xh3_curve.pdf to OUT.
    """
    beta = [0.0, 0.1, 0.5]
    series = {
        "NONE":     [482, 482, 482],
        "LOCAL":    [482, 465, 405],
        "NEIGHBOR": [479, 448, 334],
        "GLOBAL":   [484, 439, 266],
    }
    markers = {"NONE": "s", "LOCAL": "^", "NEIGHBOR": "o", "GLOBAL": "D"}

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for name, ys in series.items():
        ax.plot(beta, ys, marker=markers[name], color=C[name],
                lw=1.8, ms=5, label=name)

    # Highlight the flat NONE row = closed gate
    ax.text(0.5, 489, "NONE: flat — gate closed",
            ha="right", va="bottom", fontsize=7.5, color=C["NONE"])
    ax.annotate("$\\beta$ amplifies\nonly once sharing opens",
                xy=(0.5, 266), xytext=(0.27, 330),
                fontsize=7.5, color=C["GLOBAL"],
                arrowprops=dict(arrowstyle="->", color=C["GLOBAL"], lw=1.0))

    ax.set_xlabel("Prevention coefficient $\\beta$")
    ax.set_ylabel("Mean incidents / org-year")
    ax.set_xticks(beta)
    ax.set_ylim(230, 510)
    ax.set_title("H1$\\times$H3: sharing is the gate, $\\beta$ the amplifier")
    ax.legend(frameon=False, ncol=2, loc="lower left", handlelength=1.6)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_h1xh3_curve.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------
# Fig. 4 — H4 network-topology ranking (Table h4-ranking, exp02)
# ---------------------------------------------------------------
def fig_h4_topology():
    """Render Fig. 4: horizontal bar ranking of network topologies by incidents.

    Topologies are sorted best (fewest incidents) -> worst, with the best on top
    after inverting the y-axis. Saves fig_h4_topology.pdf to OUT.
    """
    topo  = ["Complete", "Erdős–Rényi", "Watts–Strogatz",
             "Barabási–Albert", "Star"]
    means = [273.1, 323.3, 336.0, 346.9, 382.1]
    colors = [C["GLOBAL"], C["neutral"], C["NEIGHBOR"], C["LOCAL"], C["hub"]]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    y = range(len(topo))
    bars = ax.barh(list(y), means, color=colors, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(topo)
    ax.invert_yaxis()
    for b, m in zip(bars, means):
        ax.text(m + 3, b.get_y() + b.get_height() / 2, f"{m:.1f}",
                va="center", ha="left", fontsize=8)

    ax.set_xlabel("Mean incidents / org-year")
    ax.set_xlim(0, 430)
    ax.set_title("H4: connection density beats topology family")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_h4_topology.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    fig_h1_ordering()
    fig_h1xh3_curve()
    fig_h4_topology()
    print("done.")
