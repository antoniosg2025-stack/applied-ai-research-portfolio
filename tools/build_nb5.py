"""Builds notebooks/05_vending_owner_rules_restock.ipynb."""
import nbformat as nbf
from style import STYLE_CELL, header

FNAME = "05_vending_owner_rules_restock.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md(header("05", "Vending: The \"Optimal\" Restock Plan Versus the Owner's Rules, and Why the Owner Was Right", FNAME,
    "I run four vending machines in Ontario (a car dealership, a hotel, a medical clinic an hour away, and an auto-parts yard). "
    "An agent turns the payment terminal's sales data into restock lists. I keep correcting the same things: keep reserve "
    "pieces next to the hotel machine, refill chocolates to at least 10 at the clinic and the hotel, buy only by the case, "
    "and never make a short trip to the clinic. This notebook asks whether the agent should learn those rules, "
    "and measures what each rule costs and saves.")))

cells.append(code(STYLE_CELL))
cells.append(code('''thesis_card("4 vending machines, weekly restock",
            ["Clinic is an hour away: no short trips", "Time: full-time student",
             "Buys only by the case (Walmart)", "Hotel contact refills from a reserve",
             "Chocolates sell a lot: never run short"],
            "what to restock, how much,\\nand when to drive",
            "my corrections to\\neach restock list")'''))

cells.append(md("""## 1. Literature

| Research line | Key result | What it gives this case |
|---|---|---|
| Bounded rationality [1] | Real decisions are made under limits of time, information and computation | The owner's limits (classes, a one-hour drive) belong in the decision, not outside it |
| Negative binomial retail demand [2] | Retail demand with unobserved lost sales is better described by a negative binomial than by Poisson | The demand model used below |
| Smart vending inventory and routing [3] | Replenishment points, order-up-to levels and vehicle routes must be decided jointly | The "optimal" planner the owner is compared with |
| Simple heuristics that make us smart [4] | Simple rules of thumb can match or beat complex models when information is limited | The owner's rules may be efficient heuristics, not biases |
| Learning to defer and personalization [5, 6] | Agents should learn when and how to follow a specific human | The agent should learn *these* rules from *this* owner |

**Gap.** Inventory research optimizes a stated cost function. It does not ask what to do when the owner's corrections disagree with the optimum, and whether those corrections carry costs the model does not see.
"""))

cells.append(md("""## 2. The owner's rules (all stated by me while working with the agent)

| Rule | Where | What the agent's plan does without it |
|---|---|---|
| Keep reserve pieces next to the machine (chips and drinks); the hotel contact refills empty coils from it between visits | Hotel | No reserve: empty coils stay empty until the next visit |
| Refill every chocolate coil to at least 10 | Clinic and hotel | Fills to the forecast, often less than 10 |
| Buy only by the case | All | Buys the exact units (not possible at the store) |
| No short trips: go only for a full refill | Clinic | Plans a visit every week |
"""))

cells.append(code('''import numpy as np
import pandas as pd

CATS = [("chips", 4, 8, 1.10, 40), ("chocolate", 6, 10, 1.10, 24), ("drink", 10, 5, 1.40, 24)]  # coils, capacity, margin, case size
MACH = pd.DataFrame([("dealership", 1.30, 16, [1.2, 1.2, 1.2, 1.2, 1.1, 0.5, 0.2]),
                     ("hotel", 1.20, 190, [0.8, 0.8, 0.8, 0.9, 1.2, 1.4, 1.3]),
                     ("clinic", 0.90, 220, [1.3, 1.3, 1.3, 1.3, 1.2, 0.2, 0.0]),
                     ("parts_yard", 0.45, 50, [1.2, 1.2, 1.2, 1.2, 1.2, 0.4, 0.1])],
                    columns=["machine", "base", "km", "dow"]).set_index("machine")
KM_COST, DISP = 0.70, 2.0

rng0 = np.random.default_rng(3)
coils = []
for m, r in MACH.iterrows():
    for cat, n, cap, margin, case in CATS:
        pop = rng0.lognormal(0, 0.45, n); pop /= pop.mean()
        for j in range(n):
            coils.append(dict(machine=m, kind=cat, cap=cap, margin=margin, case=case, mean=r.base * pop[j] * np.mean(r.dow)))
coils = pd.DataFrame(coils)

def nb(rng, mu): return rng.poisson(rng.gamma(DISP, mu / DISP)) if mu > 0 else 0
coils.groupby("machine").agg(coils=("kind", "size"), expected_units_per_day=("mean", "sum"),
                             round_trip_km=("machine", lambda m: MACH.loc[m.iloc[0], "km"])).round(1)'''))

cells.append(md("""## 3. Experiment 1: does the agent stop needing the same corrections?

Each week the agent writes a restock list per machine, the owner reviews it and corrects every line that breaks one of his rules, naming the rule. Two agents:

- **No memory of corrections**: the owner corrects the same lines every week (what happened with each new chat or new user).
- **Learns each correction as a rule**: after one correction, the rule is applied by the agent from then on.
"""))

cells.append(code('''RULE_NAMES = ["hotel reserve", "chocolates to 10 (clinic, hotel)", "buy by the case", "clinic: full refill only"]

def plan(week, forecast_col, rules):
    """Restock targets per coil for this week's visits, and the trips."""
    visits = {m: True for m in MACH.index}
    if "clinic: full refill only" in rules:
        visits["clinic"] = week % 2 == 0                         # full refill every two weeks
    target = {}
    for i, c in coils.iterrows():
        if not visits[c.machine]: continue
        days = 14 if (c.machine == "clinic" and "clinic: full refill only" in rules) else 7
        t = min(c.cap, int(np.ceil(c[forecast_col] * days)))
        if "chocolates to 10 (clinic, hotel)" in rules and c.kind == "chocolate" and c.machine in ("clinic", "hotel"):
            t = max(t, 10)
        target[i] = t
    reserve = {"chips": 10, "drink": 12} if "hotel reserve" in rules else {}
    return visits, target, reserve

def violations(visits, target, reserve, rules_learned, week):
    v = 0
    for i, t in target.items():
        c = coils.loc[i]
        if c.kind == "chocolate" and c.machine in ("clinic", "hotel") and t < 10: v += 1
    if not reserve: v += 1
    if "buy by the case" not in rules_learned: v += 1
    if week % 2 == 1 and visits["clinic"]: v += 1               # a short trip to the clinic
    return v

def correction_curve(learns, weeks=12):
    learned, out = set(), []
    for w in range(weeks):
        visits, target, reserve = plan(w, "mean", learned)
        v = violations(visits, target, reserve, learned, w)
        out.append(dict(week=w + 1, corrections=v))
        if learns:
            learned |= set(RULE_NAMES)                           # the owner named every broken rule this week
    return pd.DataFrame(out)

cc = pd.DataFrame({"no memory of corrections": correction_curve(False).set_index("week").corrections,
                   "learns each correction as a rule": correction_curve(True).set_index("week").corrections})
print("Total corrections the owner makes in 12 weeks:"); print(cc.sum())'''))

cells.append(code('''fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.plot(cc.index, cc["no memory of corrections"], "-o", color=PALETTE["coral"], lw=2.6, ms=8, label="no memory of corrections")
ax.plot(cc.index, cc["learns each correction as a rule"], "-o", color=PALETTE["teal"], lw=2.6, ms=8, label="learns each correction as a rule")
ax.set_xticks(cc.index); ax.set_ylim(bottom=0)
ax.set_title("Lines the owner has to correct in each week's restock list")
ax.set_xlabel("Week"); ax.set_ylabel("Corrections per week"); ax.legend()
plt.tight_layout(); plt.show()'''))

cells.append(md("""## 4. Experiment 2: what does each of the owner's rules cost and save?

12 weeks of demand are simulated with stockouts. The **optimal plan** fills each coil to its forecast demand until the next visit and visits every machine every week. Then each owner rule is added alone, and finally all of them together. Outcome: margin lost to empty coils, travel cost, and margin kept after travel (all per week).
"""))

cells.append(code('''def simulate(forecast_col, rules, weeks=12, seed=11):
    rng = np.random.default_rng(seed)
    stock = coils.cap.to_numpy().copy(); lost = sold_m = travel = 0.0
    reserve_left = {}
    for w in range(weeks):
        visits, target, reserve = plan(w, forecast_col, rules)
        for m, go in visits.items():
            if go: travel += MACH.loc[m, "km"] * KM_COST
        for i, t in target.items(): stock[i] = max(stock[i], t)
        reserve_left = dict(reserve)
        for d in range(7):
            for i, c in coils.iterrows():
                dem = nb(rng, c["mean"] / np.mean(MACH.loc[c.machine, "dow"]) * MACH.loc[c.machine, "dow"][d])
                s = min(dem, stock[i]); stock[i] -= s; sold_m += s * c.margin; lost += (dem - s) * c.margin
                if stock[i] == 0 and c.machine == "hotel" and reserve_left.get(c.kind, 0) > 0:
                    add = min(c.cap, reserve_left[c.kind]); stock[i] += add; reserve_left[c.kind] -= add
    return dict(margin_lost=lost / weeks, travel_cost=travel / weeks, margin_after_travel=(sold_m - travel) / weeks)

ALL = set(RULE_NAMES)
cases = [("optimal plan", set())] + [("+ " + r, {r}) for r in RULE_NAMES if r != "buy by the case"] + [("all owner's rules", ALL)]
money = pd.DataFrame({name: simulate("mean", r) for name, r in cases}).T.round(0)
money.columns = ["margin lost to empty coils ($/week)", "travel cost ($/week)", "margin after travel ($/week)"]
money'''))

cells.append(code('''fig, ax = plt.subplots(figsize=(10.5, 4.8))
cols = [PALETTE["grey"], PALETTE["sky"], PALETTE["sky"], PALETTE["sky"], PALETTE["teal"]]
vals = money["margin after travel ($/week)"]
bars = ax.bar([n.replace(" (clinic, hotel)", "\\n(clinic, hotel)").replace(": ", ":\\n") for n in money.index], vals, color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, f"${v:,.0f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(0, vals.max() * 1.15); ax.grid(axis="x", visible=False)
ax.set_title("Weekly margin after travel, four machines")
ax.set_ylabel("$ per week")
plt.tight_layout(); plt.show()'''))

cells.append(md("""## 5. Findings

> **Key finding 1.** Without memory, the owner corrects the same lines every week (150 corrections in 12 weeks: every chocolate coil, the reserve, the cases and the short trip); an agent that turns each correction into a rule needs them **once** (12, all in the first week). This is the re-specification cost from notebook 01, measured on a real routine.
>
> **Key finding 2.** The "optimal" plan is optimal for a cost function that leaves out the owner's situation. The rule "no short trips to the clinic" **loses a little more margin** to empty coils (about \$30 a week) but saves about \$77 a week in driving, the largest single gain. With all of the owner's rules, margin after travel goes from about \$83 to \$155 a week. The owner was trading lost sales for an hour of road each way, a trade the planner could not see because his time and distance were not in its objective.
>
> **Key finding 3.** The hotel reserve recovers about \$21 a week of sales between visits at almost no cost, and the chocolate minimum adds about \$7, because most of those coils already fill to capacity. An agent that learns the owner's rules **and** measures what each one costs and saves can tell the owner which rules carry the result, instead of obeying all of them blindly or overriding them.

**Limitations:** synthetic demand calibrated to my coil layout, illustrative distances, and case sizes that only affect the purchase list (extra units go to home stock).

## 6. Thesis questions

1. Can an agent infer *why* an owner's rule works (here, the owner's travel time) and propose to drop it when the reason no longer holds?
2. When the owner's situation changes (for example, moving to Alberta and not visiting the clinic at all), which rules should transfer to a new set of machines?
3. How should the agent present a disagreement with the owner (with evidence, in money) without overriding him?

## References

1. Simon, H. A. (1955). *A behavioral model of rational choice.* Quarterly Journal of Economics, 69(1), 99–118.
2. Agrawal, N., & Smith, S. A. (1996). *Estimating negative binomial demand for retail inventory management with unobservable lost sales.* Naval Research Logistics, 43(6), 839–861.
3. Park, Y.-B., & Yoo, J.-S. (2012). *A heuristic for the inventory management of smart vending machine systems.* Journal of Industrial Engineering and Management, 5(2), 354–363.
4. Gigerenzer, G., Todd, P. M., & the ABC Research Group (1999). *Simple Heuristics That Make Us Smart.* Oxford University Press.
5. Mozannar, H., & Sontag, D. (2020). *Consistent Estimators for Learning to Defer to an Expert.* ICML 2020.
6. Salemi, A., et al. (2024). *LaMP: When Large Language Models Meet Personalization.* ACL 2024.
"""))

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nbf.write(nb, f"notebooks/{FNAME}")
print("written")
