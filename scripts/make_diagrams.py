"""Generates the five diagrams (Fig. 1 to Fig. 5) of the thesis.
Run:  python3 make_diagrams.py        (needs matplotlib and numpy)
Output: figures/fig01_pipeline_overview.png ... figures/fig05_evaluation_protocol.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

os.makedirs("figures", exist_ok=True)
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
INK, GREY, LIGHT = "#1f2933", "#6b7280", "#eef1f5"
BLUE, TEAL, ORANGE = "#2f5d8a", "#2a7f7f", "#c8641e"
BLUE_L, TEAL_L, ORANGE_L = "#dce8f4", "#d8eeee", "#fbe6d6"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=INK, fs=10, bold=False, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1
fig, ax = canvas(11, 4.6, 22, 9)
xs = [0.3, 3.9, 7.8, 12.0, 16.0, 19.6]
ws = [3.0, 3.1, 3.7, 3.5, 3.4, 2.2]
labels = ["Passages and\nqueries\n(text)",
          "Sentence encoder\n(all-MiniLM-L6-v2,\n384 dimensions)",
          "First-stage ANN index\nHNSW, SuCo, TaCo,\nIVF-PQ, Annoy, LSH",
          "Exact distances\nto the candidates\n(top 100 returned)",
          "Cross-encoder\nre-ranking of the\n100 candidates",
          "Ranked\npassages\nfor the\ngenerator"]
fcs = [LIGHT, LIGHT, BLUE_L, LIGHT, ORANGE_L, LIGHT]
for x, w, t, fc in zip(xs, ws, labels, fcs):
    box(ax, x, 4.4, w, 2.6, t, fc=fc, fs=10)
for i in range(5):
    arrow(ax, xs[i] + ws[i] + 0.03, 5.7, xs[i + 1] - 0.03, 5.7)
ax.text(11, 8.4, "Retrieval pipeline studied", ha="center", fontsize=13, fontweight="bold", color=INK)
# additions
box(ax, 4.6, 0.3, 4.8, 1.7, "Query-adaptive budget\nhow collisions are counted", fc=TEAL_L, ec=TEAL, fs=9.5)
box(ax, 9.9, 0.3, 4.8, 1.7, "Cluster pruning\nwhich points become candidates", fc=TEAL_L, ec=TEAL, fs=9.5)
box(ax, 15.6, 0.3, 5.4, 1.7, "Cross-encoder re-ranking\nfinal ranking of the passages", fc=ORANGE_L, ec=ORANGE, fs=9.5)
arrow(ax, 7.0, 2.05, 9.0, 4.35, color=TEAL, ls="--")
arrow(ax, 12.3, 2.05, 10.8, 4.35, color=TEAL, ls="--")
arrow(ax, 18.3, 2.05, 17.7, 4.35, color=ORANGE, ls="--")
ax.text(0.3, 1.25, "Additions studied:", fontsize=10, color=GREY, style="italic", va="center")
save(fig, "fig01_pipeline_overview")

# ---------------------------------------------------------------- Fig. 2
fig, ax = canvas(11, 7.4, 22, 14.6)
ax.text(11, 14.1, "Subspace-collision framework", ha="center", fontsize=13, fontweight="bold", color=INK)
ax.text(0.3, 12.9, "Index construction (once)", fontsize=11, fontweight="bold", color=BLUE)
b = [(0.3, "Data vectors $X$\n($n$ vectors,\n$d$ dimensions)"),
     (4.9, "PCA projection and\neigenvalue-balanced\nsubspace assignment"),
     (9.5, "$N_s$ subspaces;\neach split into\ntwo halves"),
     (14.1, "k-means ($K = 50$) on\neach half: inverted\nmulti-index (IMI)"),
     (18.6, "Per subspace:\ncell → point ids\n(hash map)")]
for x, t in b:
    box(ax, x, 10.0, 3.3, 2.4, t, fs=10)
for i in range(4):
    arrow(ax, b[i][0] + 3.33, 11.2, b[i + 1][0] - 0.03, 11.2)
ax.text(0.3, 8.6, "Query processing (per query)", fontsize=11, fontweight="bold", color=BLUE)
q = [(0.3, "Query $q$,\ncentred and\nprojected", LIGHT),
     (4.2, "Distances from the\ntwo halves to the $K$\ncentroids, sorted", LIGHT),
     (8.4, "Dynamic Activation:\nvisit IMI cells until\nthe budget is reached", TEAL_L),
     (12.6, "SC-scores:\ncollision count\nof each point", LIGHT),
     (16.4, "Boundary score\nfrom the histogram\n($\\beta n$ rule)", LIGHT)]
wq = [3.3, 3.7, 3.7, 3.3, 3.5]
for (x, t, fc), w in zip(q, wq):
    box(ax, x, 5.2, w, 2.4, t, fc=fc, ec=TEAL if fc == TEAL_L else INK, fs=10)
for i in range(4):
    arrow(ax, q[i][0] + wq[i] + 0.03, 6.4, q[i + 1][0] - 0.03, 6.4)
ax.text(10.25, 8.55, "Query-adaptive budget: budgets $b_s$\nreplace $\\alpha n$ in each subspace", ha="center", fontsize=9.5, color=TEAL, style="italic")
arrow(ax, 10.25, 8.1, 10.25, 7.65, color=TEAL, ls="--")
box(ax, 9.0, 0.5, 4.4, 2.2, "Candidates\n(score ≥ boundary;\nrestricted by pruning)", fc=TEAL_L, ec=TEAL, fs=10)
box(ax, 15.2, 0.5, 4.2, 2.2, "Exact distances,\ntop-$k$ returned", fs=10)
box(ax, 0.3, 0.5, 5.6, 2.2, "k-means with $K_c$ clusters on the\noriginal vectors (at start-up)", fc=TEAL_L, ec=TEAL, fs=10)
arrow(ax, 18.1, 5.15, 12.0, 2.75)
arrow(ax, 13.43, 1.6, 15.17, 1.6)
arrow(ax, 5.93, 1.6, 8.97, 1.6, color=TEAL, ls="--")
ax.text(7.45, 2.15, "cluster pruning:\n$T_c$ nearest clusters", ha="center", fontsize=9.5, color=TEAL, style="italic")
save(fig, "fig02_subspace_collision_flow")

# ---------------------------------------------------------------- Fig. 3
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
m = np.array([0.30, 0.05, 0.22, 0.08, 0.27, 0.08])           # illustrative energies
B = 600.0
uni = np.full(6, B / 6)
raw = np.maximum(np.floor(B * m / m.sum()), 5)
ad = raw.copy(); ad[np.argmax(m)] += B - ad.sum()
for a, v, ttl, c in ((axs[0], uni, "Uniform budget (baseline)", BLUE), (axs[1], ad, "Query-adaptive budget", TEAL)):
    a.bar(np.arange(1, 7), v, color=c, width=0.65)
    a.set_title(ttl, fontsize=11, color=INK)
    a.set_xlabel("Subspace s", fontsize=10)
    a.set_xticks(range(1, 7))
    a.axhline(100, color=GREY, lw=0.8, ls="--")
    a.spines[["top", "right"]].set_visible(False)
    for i, val in enumerate(v):
        a.text(i + 1, val + 6, f"{int(val)}", ha="center", fontsize=9, color=INK)
axs[0].set_ylabel("Points to collect in the subspace", fontsize=10)
axs[0].set_ylim(0, max(ad) * 1.18)
axs[1].text(6.55, 106, "α·n", fontsize=9, color=GREY, ha="right")
fig.text(0.5, -0.04, "Illustrative values for one query; both panels distribute the same total budget $N_s\\,\\alpha n$ over six subspaces.",
         ha="center", fontsize=9, color=GREY, style="italic")
fig.tight_layout()
save(fig, "fig03_query_adaptive_budget")

# ---------------------------------------------------------------- Fig. 4
rng = np.random.default_rng(7)
cent = np.array([[1.5, 1.5], [4.5, 1.2], [7.5, 1.8], [1.8, 4.6], [4.8, 4.8], [7.8, 4.7], [3.2, 7.4], [6.5, 7.5]])
pts, lab = [], []
for k, c in enumerate(cent):
    p = c + rng.normal(0, 0.55, (40, 2)); pts.append(p); lab += [k] * 40
pts = np.vstack(pts); lab = np.array(lab)
qpt = np.array([5.2, 4.1])
d = np.linalg.norm(cent - qpt, axis=1); probe = np.argsort(d)[:2]
fig, ax = plt.subplots(figsize=(6.6, 5.4))
for k in range(len(cent)):
    sel = lab == k
    inside = k in probe
    ax.scatter(pts[sel, 0], pts[sel, 1], s=16, color=TEAL if inside else "#b8bec7", zorder=2)
    ax.add_patch(Circle(cent[k], 1.35, fill=inside, fc=TEAL_L if inside else "none", ec=TEAL if inside else "#9aa1ab",
                        lw=1.4 if inside else 0.9, ls="-" if inside else "--", alpha=0.55 if inside else 1, zorder=1))
    ax.scatter(*cent[k], marker="x", color=INK, s=40, zorder=3)
ax.scatter(*qpt, marker="*", s=240, color=ORANGE, ec=INK, zorder=4)
ax.annotate("query", qpt, xytext=(qpt[0] + 0.5, qpt[1] - 1.1), fontsize=10, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE))
ax.scatter([], [], s=16, color=TEAL, label="points in probed clusters (candidates)")
ax.scatter([], [], s=16, color="#b8bec7", label="points in other clusters (skipped)")
ax.scatter([], [], marker="x", color=INK, label="cluster centroids ($K_c = 8$, $T_c = 2$ here)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=False)
ax.set_xlim(-0.8, 9.6); ax.set_ylim(-0.8, 9.6); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Cluster pruning (schematic, two dimensions)", fontsize=11, color=INK)
save(fig, "fig04_cluster_pruning")

# ---------------------------------------------------------------- Fig. 5
fig, ax = canvas(11, 5.6, 22, 11.2)
ax.text(11, 10.6, "Evaluation protocol", ha="center", fontsize=13, fontweight="bold", color=INK)
top = [(0.3, "Datasets\nDeep96 (100K–2M), SIFT-1M,\nGloVe, NYTimes, Gist (250K–1M)", BLUE_L),
       (7.6, "Ground truth\nexact 50 nearest neighbours\n(brute force)", LIGHT),
       (14.5, "Methods and settings\nHNSW, IVF-PQ, Annoy, LSH,\nSuCo, TaCo, additions", LIGHT)]
for x, t, fc in top:
    box(ax, x, 6.6, 6.2 if x < 14 else 7.2, 2.7, t, fc=fc, fs=11)
arrow(ax, 6.53, 7.95, 7.57, 7.95); arrow(ax, 13.83, 7.95, 14.47, 7.95)
mid = [(0.3, "Measurement\none query thread, 1,000 queries\nrecall@50 against query time", TEAL_L),
       (7.6, "Matched recall\nsettings compared at the\nsame recall level", TEAL_L),
       (14.5, "Statistics\nmedians of repeated runs;\npaired bootstrap, 95% intervals", TEAL_L)]
for x, t, fc in mid:
    box(ax, x, 2.9, 6.2 if x < 14 else 7.2, 2.7, t, fc=fc, ec=TEAL, fs=11)
arrow(ax, 11.0, 6.55, 3.5, 5.65, ls="--", color=GREY)
arrow(ax, 6.53, 4.25, 7.57, 4.25); arrow(ax, 13.83, 4.25, 14.47, 4.25)
box(ax, 4.0, 0.2, 14.0, 1.9, "Retrieval-augmented generation: MS MARCO and FiQA passages → first stage → cross-encoder\nnDCG@10, MRR@10, hit rates, Recall@100, time per stage", fc=ORANGE_L, ec=ORANGE, fs=11)
arrow(ax, 11.0, 2.85, 11.0, 2.15, color=ORANGE)
save(fig, "fig05_evaluation_protocol")
print("done")
