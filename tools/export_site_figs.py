"""Runs notebooks 02-07 and extracts every chart's data (lines, bars, heatmaps) to site/data/figs.json."""
import json, sys, os, io, contextlib, nbformat, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

def ax_data(ax):
    if not ax.axison: return None
    d = dict(title=ax.get_title(loc="left") or ax.get_title(), xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(),
             xlog=ax.get_xscale()=="log", ylog=ax.get_yscale()=="log", lines=[], bars=[], heat=None, hbars=[])
    xt = [t.get_text() for t in ax.get_xticklabels()]
    yt = [t.get_text() for t in ax.get_yticklabels()]
    for ln in ax.get_lines():
        x, y = ln.get_xdata(), ln.get_ydata()
        try: x = np.asarray(x, float); y = np.asarray(y, float)
        except Exception: continue
        if len(x) < 2: continue
        lab = ln.get_label()
        if lab.startswith("_"):
            # skip reference lines (axvline/axhline) but keep them as markers
            if len(set(np.round(x, 6))) == 1: d.setdefault("vlines", []).append(float(x[0])); continue
            if len(set(np.round(y, 6))) == 1: d.setdefault("hlines", []).append(float(y[0])); continue
        d["lines"].append(dict(label=lab if not lab.startswith("_") else "", x=np.round(x, 4).tolist(), y=np.round(y, 4).tolist(),
                               color=to_hex(ln.get_color()), dash=ln.get_linestyle() not in ("-", "solid")))
    for c in ax.containers:
        if not hasattr(c, "patches") or not c.patches: continue
        ps = c.patches
        horiz = getattr(c, "orientation", "vertical") == "horizontal"
        if horiz:
            d["hbars"].append(dict(label=c.get_label() if not str(c.get_label()).startswith("_") else "",
                                   y=[p.get_y() + p.get_height()/2 for p in ps], w=[p.get_width() for p in ps], l=[p.get_x() for p in ps],
                                   color=[to_hex(p.get_facecolor()) for p in ps]))
        else:
            d["bars"].append(dict(label=c.get_label() if not str(c.get_label()).startswith("_") else "",
                                  x=[p.get_x() + p.get_width()/2 for p in ps], h=[p.get_height() for p in ps],
                                  b=[p.get_y() for p in ps], width=ps[0].get_width(), color=[to_hex(p.get_facecolor()) for p in ps]))
    d["xticks"] = [[float(v), t] for v, t in zip(ax.get_xticks(), xt) if t]
    d["yticks"] = [[float(v), t] for v, t in zip(ax.get_yticks(), yt) if t]
    if ax.images:
        im = ax.images[0]; arr = np.asarray(im.get_array(), float)
        d["heat"] = dict(z=np.round(arr, 3).tolist(), xt=xt, yt=yt, vmin=float(im.norm.vmin), vmax=float(im.norm.vmax), cmap=im.get_cmap().name,
                         text=[[t.get_text() for t in ax.texts]])
    if not (d["lines"] or d["bars"] or d["hbars"] or d["heat"]): return None
    return d

out = {}
for nbf in sorted(os.listdir("notebooks"))[1:]:
    key = nbf[:2]; nb = nbformat.read(f"notebooks/{nbf}", 4)
    figs = []; prints = []
    def show(*a, **k):
        for f in plt.get_fignums():
            fig = plt.figure(f)
            got = [dd for dd in (ax_data(ax) for ax in fig.axes) if dd]
            if got:
                main = got[0]
                for extra in got[1:]:
                    if not extra["title"]:
                        main["y2"] = extra; continue
                    figs.append(extra)
                figs.insert(len(figs) - (len(got) - 1 - sum(1 for e in got[1:] if not e["title"])), main) if False else None
                figs.append(main) if main not in figs else None
        plt.close("all")
    g = {"__name__": "__main__"}
    for c in nb.cells:
        if c.cell_type != "code": continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(c.source.replace("plt.show()", "_show()"), g) if "_show" in g else None
            if "_show" not in g:
                exec(c.source, g)
                if "plt" in g: g["_show"] = show; g["plt"].show = show; show()
        prints.append(buf.getvalue())
    out[key] = figs
    if key == "03":
        w = g["wait"]; out["press_wait"] = w.reset_index().to_dict("records")
    if key == "04":
        out["seo_log_counts"] = g["log"].status.value_counts().to_dict(); out["seo_log_total"] = int(len(g["log"]))
        out["seo_policy"] = g["res"].reset_index().to_dict("records")
    print(key, len(figs), [f["title"][:60] for f in figs])
json.dump(out, open("site/data/figs.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
