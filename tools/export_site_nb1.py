"""Exports the data behind case 01 for the interactive site (site/data/case01.json)."""
import json, nbformat, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
nb = nbformat.read("notebooks/01_owner_decision_profile.ipynb", 4)
g = {}
for c in nb.cells:
    if c.cell_type == "code" and "plt.subplots" not in c.source and "thesis_card(" not in c.source:
        exec(c.source, g)
runs, summary, spec, W = g["runs"], g["summary"], g["spec"], g["W"]
out = {}
out["weights"] = {"types": list(W.index), "features": list(W.columns), "values": W.values.round(2).tolist()}
out["summary"] = summary.reset_index().rename(columns={"index": "agent"}).to_dict("records")
days = list(range(120))
out["asks_per_day"] = {n: (r.groupby("day").asked.mean().rolling(7, min_periods=1).mean() * 40).round(2).tolist() for n, r in runs.items() if n != "fixed rules"}
out["errors_per_day"] = {n: (r.groupby("day").error.mean().rolling(7, min_periods=1).mean() * 100).round(2).tolist() for n, r in runs.items()}
out["specifier"] = {lab: (r.groupby("day").error.mean().rolling(7, min_periods=1).mean() * 100).round(2).tolist() for lab, r in spec.items()}
by_type = pd.DataFrame({n: r[r.asked == 0].groupby("type").error.mean() * 100 for n, r in runs.items() if n != "fixed rules"}).loc[g["TYPES"]]
out["by_type"] = {"types": list(by_type.index), "series": {c: by_type[c].round(2).tolist() for c in by_type}}
# Confidence threshold sweep: how errors and questions trade off
Agent, run = g["Agent"], g["run"]
sweep = []
for tau in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    for mode in ["one_profile", "per_type"]:
        r = pd.concat([run(Agent(mode, tau=tau), seed=s) for s in range(3)])
        sweep.append(dict(tau=tau, agent=mode, errors=round(100 * r.error.mean(), 2), asks=round(100 * r.asked.mean(), 2)))
        print(tau, mode, sweep[-1])
out["sweep"] = sweep
json.dump(out, open("site/data/case01.json", "w"))
print(summary)
