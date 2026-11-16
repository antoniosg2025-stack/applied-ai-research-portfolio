"""Shared pieces for the notebook builders."""
from pathlib import Path

STYLE_CELL = (Path(__file__).parent / "style_cell.py").read_text()

REPO = "antoniosg2025-stack/applied-ai-research-portfolio"

THESIS_MD = """> **Main thesis question.** How can LLM agents learn, from an owner's approvals, rejections and corrections, a **decision profile** grounded in the owner's personality and situation (constraints and strengths: economic, skills, cognitive, family and business), so that they **minimize the gap between the agent's decision and the decision the owner would have made**, across different types of decisions, as the owner evolves, and when other users operate the system on the owner's behalf, **deferring to the owner** whenever that gap is likely to be large?
>
> **Error, in this portfolio, always means:** the agent decided differently from how the owner would have decided."""


def header(num, title, fname, case_line):
    return f"""# {num} · {title}

**Juan Antonio Sainz Gálvez** · Applied AI & ML, Conestoga College · [Portfolio home](../README.md)

<a href="https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{fname}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

{THESIS_MD}

**This case:** {case_line}
"""
