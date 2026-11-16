"""Draws the README infographics into assets/."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
exec((Path(__file__).parent / "style_cell.py").read_text())
from matplotlib.patches import Circle

out = Path(__file__).parent.parent / "assets"; out.mkdir(exist_ok=True)

# 1. Thesis map: one question, seven real cases
cases = [
    ("01 Agents", "One profile per decision type;\nspecifier's answers as comments", PALETTE["navy"]),
    ("02 Equilibra", "Learn the rule over the situation,\nnot the owner's moves", PALETTE["teal"]),
    ("03 Press", "A correction with a reason\nteaches ~4x faster than yes/no", PALETTE["sky"]),
    ("04 SEO", "Autonomy by reversibility;\nrules that travel across sites", PALETTE["gold"]),
    ("05 Vending", "The owner's rules beat the\n'optimal' plan: time and distance", PALETTE["coral"]),
    ("06 GrabGo", "Camera or customer photo:\nchosen by the owner's weights", PALETTE["teal"]),
    ("07 Suppliers", "Photos, videos and a scale;\ntolerance learned per product", PALETTE["sky"]),
]
fig, ax = plt.subplots(figsize=(14, 8.2)); ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off")
ax.add_patch(Circle((7, 4.1), 1.75, color=PALETTE["navy"]))
ax.text(7, 4.55, "Decide like", ha="center", color="white", fontsize=17, fontweight="bold")
ax.text(7, 4.0, "the owner", ha="center", color="white", fontsize=17, fontweight="bold")
ax.text(7, 3.4, "and ask when the gap\nis likely to be large", ha="center", color="#C9D3E0", fontsize=10.5)
import numpy as np
angles = np.linspace(90, 90 - 360, len(cases), endpoint=False)
for (title, finding, color), a in zip(cases, np.radians(angles)):
    cx, cy = 7 + 4.9 * np.cos(a), 4.1 + 3.05 * np.sin(a)
    box(ax, (cx - 1.65, cy - 0.62), 3.3, 1.24, "", color)
    ax.text(cx, cy + 0.28, title, ha="center", color="white", fontsize=12.5, fontweight="bold")
    ax.text(cx, cy - 0.2, finding, ha="center", va="center", color="white", fontsize=9)
    ax.plot([7 + 1.8 * np.cos(a), cx - 1.0 * np.cos(a)], [4.1 + 1.8 * np.sin(a), cy - 0.75 * np.sin(a)],
            color=PALETTE["grey"], lw=1.5, ls=":", zorder=0)
ax.set_title("One thesis question, seven real businesses", fontsize=18, pad=6)
fig.savefig(out / "thesis_map.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 2. The learning loop every notebook measures
fig, ax = plt.subplots(figsize=(14, 4.2)); ax.set_xlim(0, 14); ax.set_ylim(0, 4.2); ax.axis("off")
steps = [("Owner's situation\n(limits + strengths)", PALETTE["navy"]),
         ("Agent predicts\nthe owner's decision", PALETTE["teal"]),
         ("Likely the same\n->  ACT", PALETTE["gold"]),
         ("Unsure or different\n->  ASK THE OWNER", PALETTE["coral"]),
         ("Approval, rejection\nor correction", PALETTE["sky"])]
xs = [0.2, 3.0, 5.8, 8.6, 11.4]
for (t, c), x in zip(steps, xs):
    box(ax, (x, 2.0), 2.5, 1.3, t, c, fontsize=10)
for x in xs[:-1]:
    arrow(ax, (x + 2.52, 2.65), (x + 2.78, 2.65))
ax.annotate("", xy=(4.25, 1.95), xytext=(12.65, 1.95),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["grey"], lw=2, connectionstyle="arc3,rad=-0.18"))
ax.text(8.45, 1.0, "updates the owner's profile", ha="center", fontsize=10.5, color=PALETTE["grey"], style="italic")
ax.text(8.4, 0.25, "Error = the agent decided differently from how the owner would have decided",
        ha="center", fontsize=12, color=PALETTE["navy"], fontweight="bold")
ax.set_title("The loop every notebook measures", fontsize=16, pad=4)
fig.savefig(out / "decision_loop.png", dpi=160, bbox_inches="tight"); plt.close(fig)
print("assets written:", [p.name for p in out.iterdir()])
