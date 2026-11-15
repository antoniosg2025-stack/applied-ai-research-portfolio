# Applied AI Research Portfolio

**Juan Antonio Sainz Gálvez** · Computer Systems Engineer (Tecnológico de Monterrey) · Applied AI & Machine Learning, Conestoga College

I build and operate small businesses in Canada and Mexico — vending machines, a phone self-checkout retail system, a portfolio-analysis AI connector and content channels — that are run day to day by a team of LLM agents and human specifiers. Before that, I spent 13 years in enterprise data migration (SAP) and founded an LED lighting distributor with offices in Hong Kong and Shenzhen.

**Research interest:** trustworthy AI agents for physical businesses (retail, vending and supply chain) that learn from real outcomes and know when to ask a human.

Each notebook connects one real project to current academic research, runs a reproducible experiment, and ends with open thesis questions. Every notebook opens in Google Colab with one click.

| # | Notebook | Research area | Colab |
|---|---|---|---|
| 01 | [Earned autonomy for LLM agents in a real multi-business operation](notebooks/01_human_agent_coordination.ipynb) | Multi-agent LLM systems, human-in-the-loop oversight, graduated autonomy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antoniosg2025-stack/applied-ai-research-portfolio/blob/main/notebooks/01_human_agent_coordination.ipynb) |
| 02 | [Phone self-checkout with computer vision: how many humans does an "automatic" store need?](notebooks/02_vision_self_checkout.ipynb) | Retail product recognition, vision-language embeddings, active learning with noisy corrections | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antoniosg2025-stack/applied-ai-research-portfolio/blob/main/notebooks/02_vision_self_checkout.ipynb) |
| 03 | [When should the truck go? Forecasting and replenishment for a small vending fleet with censored demand](notebooks/03_vending_fleet_forecasting.ipynb) | Intermittent and censored demand, global gradient boosting, inventory routing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antoniosg2025-stack/applied-ai-research-portfolio/blob/main/notebooks/03_vending_fleet_forecasting.ipynb) |
| 04 | [Market balances: finding co-moving groups and counterweights, and how much history they need](notebooks/04_market_balances.ipynb) | Random matrix theory, hierarchical clustering, covariance shrinkage, crisis correlations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antoniosg2025-stack/applied-ai-research-portfolio/blob/main/notebooks/04_market_balances.ipynb) |
| 05 | AI in a China–Canada–Mexico equipment supply chain (in progress) | Supplier selection, visual quality inspection | — |

## Run locally

```
pip install -r requirements.txt
jupyter notebook notebooks/
```

Data used here is public or anonymized. No customer data, keys or private code from my companies is included.
