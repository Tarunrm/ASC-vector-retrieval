#!/usr/bin/env python3
"""
Renames the three additions inside the figure scripts, re-runs them, and copies
the new PNGs to the file names used by the placeholders in the thesis document.

    Improvement 1  ->  Query-adaptive budget
    Improvement 2  ->  Cluster pruning
    Improvement 3  ->  Cross-encoder re-ranking
    Improvements 1+2 / Imp1+2  ->  Budget + pruning

Usage (WSL):   python3 rename_figure_labels.py
Backups of the original scripts are kept as *.bak. Nothing outside string labels that
start with a capital "Imp" is touched, so file names such as taco_imp1 stay as they are.
"""
import os, re, shutil, subprocess, sys

HOME = os.path.expanduser("~")
FIGDIR = os.path.join(HOME, "thesis_figs")
NAMES = ["make_figs.py", "families_fig.py"]
RULES = [
    (r"Improvements?\s*1\s*(?:\+|&|and)\s*2", "Budget + pruning"),
    (r"\bImp(?:rovement)?s?\.?\s*1\s*\+\s*2", "Budget + pruning"),
    (r"\bImprovement\s*1\b", "Query-adaptive budget"),
    (r"\bImprovement\s*2\b", "Cluster pruning"),
    (r"\bImprovement\s*3\b", "Cross-encoder re-ranking"),
    (r"\bImp\.?\s?1\b", "Query-adaptive budget"),
    (r"\bImp\.?\s?2\b", "Cluster pruning"),
    (r"\bImp\.?\s?3\b", "Cross-encoder re-ranking"),
]
# old file name (without .png)  ->  new file name used in the thesis document
RENAME = {
    "fig_5_4_ablation": "fig06_ablation_changes",
    "fig_5_6_miss_by_rank": "fig07_miss_by_rank",
    "fig_5_5_build": "fig08_build_cost",
    "fig_5_7_imp1_gist": "fig09_budget_gist",
    "fig_5_3_gist": "fig10_hnsw_gist_scaling",
    "fig_5_1_curves_deep96_2m": "fig11_curves_deep96_2m",
    "fig_5_2_curves_sift1m": "fig12_curves_sift1m",
    "fig_5_8_end_to_end": "fig13_end_to_end_ndcg",
    "fig_5_9_families": "fig14_families_curves",
}


def find(name):
    for d in (FIGDIR, HOME, os.getcwd()):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def main():
    for name in NAMES:
        path = find(name)
        if not path:
            print(f"!! {name} not found in {FIGDIR}, {HOME} or the current folder")
            continue
        src = open(path, encoding="utf-8").read()
        new = src
        for pat, rep in RULES:
            new = re.sub(pat, rep, new)
        if new != src:
            if not os.path.exists(path + ".bak"):
                shutil.copy(path, path + ".bak")
            open(path, "w", encoding="utf-8").write(new)
        print(f"{name}: {sum(a != b for a, b in zip(src.splitlines(), new.splitlines()))} line(s) changed")
        left = [(i + 1, l.strip()) for i, l in enumerate(new.splitlines())
                if re.search(r"improv|\bimp\s?[123]\b|Imp[0-9]", l) and "taco_imp" not in l]
        for i, l in left:
            print(f"   check line {i}: {l[:110]}")
        r = subprocess.run([sys.executable, path], cwd=os.path.dirname(path), capture_output=True, text=True)
        print(r.stdout[-400:] + (r.stderr[-400:] if r.returncode else ""))
        if r.returncode:
            print(f"!! {name} failed; fix the error above and run this script again")
    out = os.path.join(FIGDIR, "for_thesis")
    os.makedirs(out, exist_ok=True)
    for old, new in RENAME.items():
        p = os.path.join(FIGDIR, old + ".png")
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, new + ".png"))
            print(f"{old}.png -> for_thesis/{new}.png")
        else:
            print(f"?? {old}.png not found")
    print("\nCheck that no figure still says 'Improvement':  grep -il improvement", FIGDIR + "/*.py")


if __name__ == "__main__":
    main()
