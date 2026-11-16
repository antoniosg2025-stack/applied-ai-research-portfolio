"""Builds notebooks/04_seo_reversibility_rule_transfer.ipynb."""
import nbformat as nbf
from style import STYLE_CELL, header

FNAME = "04_seo_reversibility_rule_transfer.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md(header("04", "SEO: Which Changes Can an Agent Make Alone? Reversibility, and Owner Rules That Travel Across Projects", FNAME,
    "the SEO agent owns the search and AI visibility of three of my sites (buenavyda.com for GrabGo and Equilibra, and tepeyacrosarios.com). "
    "It writes strategies, edits a staging copy, and proposes production changes; I approve. When I correct it on one site "
    "(\"don't use that phrase\", \"never claim to be official\"), the question is whether it applies the correction to the other sites "
    "by itself, and whether it can tell a rule that is mine everywhere from a rule that belongs to one brand.")))

cells.append(code(STYLE_CELL))
cells.append(code('''thesis_card("SEO agent, 3 websites",
            ["Goal #1: be found by AIs, #2: web search", "Rule: production only with approval",
             "Rules per brand (faith, finance)", "Never claim to be official",
             "Time: approvals arrive days later"],
            "act alone, act on staging,\\nor ask before production",
            "approvals, corrections,\\nand which rules are global")'''))

cells.append(md("""## 1. Real data: the SEO agent's own log (4–10 Nov 2026)

Every line of the agent's log was classified by the kind of change and what happened to it. The table is the full log of its first week.
"""))

cells.append(code(f'''import numpy as np
import pandas as pd

import io
DATA = {{
    "seo_agent_decision_log.csv": """date,project,kind,status,owner_correction
2026-11-10,grabgo,internal document or audit,done by agent,0
2026-11-09,tepeyac,production site change,approved by owner,0
2026-11-09,tepeyac,staging change,waiting for owner,1
2026-11-09,tepeyac,production site change,waiting for owner,0
2026-11-09,tepeyac,staging change,done by agent,1
2026-11-09,tepeyac,production site change,waiting for owner,0
2026-11-09,tepeyac,internal document or audit,approved by owner,0
2026-11-08,tepeyac,internal document or audit,waiting for owner,0
2026-11-08,tepeyac,internal document or audit,done by agent,0
2026-11-05,all,staging change,waiting for owner,0
2026-11-05,all,site code or config,done by agent,0
2026-11-05,all,site code or config,done by agent,1
2026-11-05,home + grabgo,staging change,waiting for owner,0
2026-11-05,all,internal document or audit,done by agent,1
2026-11-05,all,staging change,done by agent,0
2026-11-04,grabgo,internal document or audit,waiting for owner,0
2026-11-04,grabgo,site code or config,waiting for owner,0
2026-11-04,home + grabgo,site code or config,done by agent,0
2026-11-04,all,site code or config,waiting for owner,0
2026-11-04,all,site code or config,approved by owner,0
2026-11-04,all,internal document or audit,done by agent,0
2026-11-04,all,internal document or audit,done by agent,0
2026-11-04,all,internal document or audit,done by agent,0
2026-11-04,all,internal document or audit,done by agent,0
""",
}}
def load(name):
    """The data travels inside the notebook, so it runs anywhere (Colab, Drive, GitHub)."""
    return pd.read_csv(io.StringIO(DATA[name]))

log = load("seo_agent_decision_log.csv")
pd.crosstab(log.kind, log.status, margins=True, margins_name="total")'''))

cells.append(code('''order = ["internal document or audit", "staging change", "site code or config", "production site change"]
ct = pd.crosstab(log.kind, log.status).reindex(order).fillna(0)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
left = np.zeros(len(ct))
for col, c in [("done by agent", PALETTE["teal"]), ("approved by owner", PALETTE["gold"]), ("waiting for owner", PALETTE["coral"])]:
    v = ct[col].values if col in ct else np.zeros(len(ct))
    ax.barh(ct.index, v, left=left, color=c, label=col); left += v
ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_title("SEO agent, first week: the riskier the change, the more it waits for the owner")
ax.set_xlabel("Number of changes in the log"); ax.legend(loc="lower right")
plt.tight_layout(); plt.show()
print(f"{(log.status == 'waiting for owner').mean():.0%} of the logged changes ended the week waiting for the owner; "
      f"{log.owner_correction.sum()} lines record an owner correction or rule.")'''))

cells.append(md("""Two of the corrections are rules that matter for the thesis: on Tepeyac I asked the agent not to use the phrase *«¿No estoy yo aquí, que soy tu madre?»*, and to remove the disclaimer that the site is not the official Basílica site while never claiming to be official. The first is a **brand rule** (it does not apply to a finance site); the second contains a **global rule** (never claim to be official, on any site). An agent that cannot tell them apart either forgets the rule on the next site or applies it where it does not belong.

## 2. Literature

| Research line | Key result | What it gives this case |
|---|---|---|
| Concrete problems in AI safety [1] | Avoiding negative side effects and safe exploration are core problems for agents that act in the world | Production changes are side effects the owner cannot easily undo |
| Side effects and reversibility [2] | Penalizing actions that reduce the reachability of states (irreversibility) yields agents that avoid unrecoverable changes | Reversibility is a principled axis for granting autonomy |
| Written principles applied broadly (Constitutional AI) [3] | A short list of natural-language principles can steer a model's behaviour across many situations | An owner's correction can become a principle reused everywhere |
| Generative Engine Optimization [4] | Specific content strategies raise visibility in generative engines by up to 40%, keyword stuffing does not help, and what works varies by domain | The owner's goal (be found by AIs) is domain-specific, so priorities differ per site |
| Learning to defer [5] | Learn when to pass the decision to the expert | Asking is a decision with a cost, not a default |

**Gap.** Safety research treats reversibility in physical environments; personalization research treats preferences one task at a time. The thesis needs both: **which actions** an agent may take alone, and **how far** an owner's correction should travel.
"""))

cells.append(md("""## 3. Experiment: four agents over 12 weeks and three sites

Each week the agent produces 30 changes spread over the three sites, of four kinds (internal document, staging change, site code or config, production change). The owner has **3 global rules** (true on every site) and **2 brand rules** (true on one site only). Each change may touch a rule. The owner rejects any change that breaks a rule that applies to that site, and answers questions once a week (a question waits 3.5 days on average).

1. **Ask for everything** (how the agent started).
2. **Autonomy by reversibility**: acts alone on internal documents and staging; asks before code and production.
3. **Reversibility + rules learned per site**: when corrected, the agent remembers the rule *for that site only*.
4. **Reversibility + rules that travel**: when corrected, the agent asks once "does this apply to all sites?" and stores the rule as global or brand-specific.

**Error** = a change the agent made that the owner would have rejected. **Re-specification** = the owner has to state a rule he already stated.
"""))

cells.append(code('''SITES = ["buenavyda", "grabgo", "tepeyac"]
KINDS = {"internal document": 0.35, "staging change": 0.30, "site code or config": 0.15, "production change": 0.20}
REVERSIBLE = {"internal document", "staging change"}
RULES = {"G1": "all", "G2": "all", "G3": "all", "B_tepeyac_phrase": "tepeyac", "B_finance_disclaimer": "buenavyda"}

def applies(rule, site): return RULES[rule] == "all" or RULES[rule] == site

def week_changes(rng, n=30):
    for _ in range(n):
        site = SITES[rng.integers(3)]; kind = rng.choice(list(KINDS), p=list(KINDS.values()))
        touched = [r for r in RULES if rng.random() < 0.06]          # rules this change could break
        yield site, kind, touched

def run(policy, weeks=12, seed=0):
    rng = np.random.default_rng(seed)
    known = {}                     # rule -> "all" | set of sites it is known for
    stated = {r: 0 for r in RULES} # how many times the owner had to state each rule
    q = err = act = 0; delay = []
    for w in range(weeks):
        for site, kind, touched in week_changes(rng):
            breaks = [r for r in touched if applies(r, site)]
            blocks = [r for r in touched if r in known and (known[r] == "all" or site in known[r])]
            if blocks and policy != "ask everything":
                act += 1; continue                         # agent avoids the change by itself
            ask = policy == "ask everything" or kind not in REVERSIBLE
            if ask:
                q += 1; delay.append(rng.uniform(0, 7))
                for r in breaks:                           # the owner rejects and explains the rule
                    stated[r] += 1
                    if policy == "rules travel":
                        known[r] = "all" if RULES[r] == "all" else known.get(r, set()) | {site}
                    elif policy == "rules per site":
                        known[r] = known.get(r, set()) | {site} if known.get(r) != "all" else "all"
            else:
                act += 1
                if breaks:
                    err += 1                                # made alone, the owner would have rejected it
                    for r in breaks:                       # discovered later, corrected
                        stated[r] += 1
                        if policy == "rules travel":
                            known[r] = "all" if RULES[r] == "all" else known.get(r, set()) | {site}
                        elif policy == "rules per site":
                            known[r] = known.get(r, set()) | {site}
    return dict(policy=policy, questions_to_owner=q, avg_days_waiting=np.mean(delay) if delay else 0,
                changes_owner_would_reject=err, re_specifications=sum(max(0, v - 1) for v in stated.values()))

POL = ["ask everything", "reversibility", "rules per site", "rules travel"]
res = pd.DataFrame([run(p, seed=s) for p in POL for s in range(20)]).groupby("policy").mean().loc[POL].round(1)
res["owner_days_lost_waiting"] = (res.questions_to_owner * res.avg_days_waiting).round(0)
res'''))

cells.append(code('''for col, title, c in [("re_specifications", "Times the owner repeats a rule he already gave (12 weeks)", PALETTE["coral"]),
                      ("changes_owner_would_reject", "Changes made alone that the owner would have rejected (12 weeks)", PALETTE["gold"])]:
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    bars = ax.bar(POL, res[col], color=c); ax.grid(axis="x", visible=False)
    for b_, v in zip(bars, res[col]):
        ax.text(b_.get_x() + b_.get_width() / 2, v + res[col].max() * 0.02, f"{v:.1f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_title(title); ax.set_ylabel("Count")
    plt.tight_layout(); plt.show()'''))

cells.append(md("""### What if the agent assumes every correction is global?

A tempting shortcut is to apply every correction everywhere. For brand rules this is wrong: the Tepeyac phrase rule would block legitimate changes on the finance site. The cell below counts those **wrongly blocked** changes.
"""))

cells.append(code('''def overgeneralize(weeks=12, seed=0):
    rng = np.random.default_rng(seed); known = set(); wrong_blocks = 0
    for w in range(weeks):
        for site, kind, touched in week_changes(rng):
            if any(r in known for r in touched) and not any(applies(r, site) for r in touched if r in known):
                wrong_blocks += 1
            for r in touched:
                if applies(r, site): known.add(r)
    return wrong_blocks
print(f"Treating every correction as global wrongly blocks about {np.mean([overgeneralize(seed=s) for s in range(20)]):.1f} "
      f"good changes in 12 weeks; asking once 'does this apply to all sites?' blocks none.")'''))

cells.append(md("""## 4. Findings

> **Key finding 1 (real data).** In its first week the SEO agent finished 7 of 10 internal documents by itself, but 3 of 5 staging changes and 2 of 3 production changes ended the week **waiting for the owner** (38% of everything it logged). The owner's approval, not the agent's work, set the pace.
>
> **Key finding 2.** Granting autonomy by **reversibility** (act alone on documents and staging, ask before code and production) cuts the owner's questions from 360 to about 126 in 12 weeks, but the agent then makes about 44 reversible changes the owner would reject, and the owner keeps repeating the same rules (about 70 times).
>
> **Key finding 3.** Corrections that **travel** (asked once whether they apply everywhere) bring errors down to about 3 and repeated rules to **zero**, against about 6 errors and 6 repeats when rules are learned site by site. Treating every correction as global instead would wrongly block about 21 good changes. The owner's profile has a **scope**: some rules are his everywhere, some belong to one brand, and the cheapest way to learn the scope is one question at the moment of the correction.

**Limitations:** a one-week real log (24 entries), simulated rules and frequencies, and an owner who always explains his rejections.

## 5. Thesis questions

1. Can an LLM agent infer a rule's scope (global, brand, page) from the wording of the correction and the context, and ask only when unsure?
2. Is reversibility enough, or does the owner's profile also weigh visibility (a reversible change seen by many clients may be worse than an irreversible internal one)?
3. How should rules learned by one agent (SEO) reach other agents (Press, GrabGo) that serve the same owner?

## References

1. Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). *Concrete Problems in AI Safety.* arXiv:1606.06565.
2. Krakovna, V., Orseau, L., Kumar, R., Martic, M., & Legg, S. (2019). *Penalizing side effects using stepwise relative reachability.* arXiv:1806.01186.
3. Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* arXiv:2212.08073.
4. Aggarwal, P., Murahari, V., Rajpurohit, T., Kalyan, A., Narasimhan, K., & Deshpande, A. (2024). *GEO: Generative Engine Optimization.* KDD 2024. arXiv:2311.09735.
5. Mozannar, H., & Sontag, D. (2020). *Consistent Estimators for Learning to Defer to an Expert.* ICML 2020.
"""))

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nbf.write(nb, f"notebooks/{FNAME}")
print("written")
