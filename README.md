# Supplementary Material

Three-notebook companion to *Post-Match Outcome Prediction in Soccer: Leveraging Shot-level xG and the Poisson-Binomial Distribution*, reproducing the full pipeline end-to-end on the five 2015/16 European top leagues. The analytical sample comprises **1822 matches** with complete records: both StatsBomb shot-level xG data and Bet365 pre-match odds. StatsBomb provides 377 of the 380 Ligue 1 matches (3 are absent from the open data), and one of those 377 cannot be matched to a Bet365 odds entry, leaving 376 complete Ligue 1 records. The remaining four leagues have full coverage (380 + 306 + 380 + 380 = 1446 matches). Notebook 0 enforces this filter at save time.

## Contents

| File | Purpose |
|---|---|
| `00_data_acquisition.ipynb`       | Download StatsBomb shot events for the five leagues, merge in football-data.co.uk pre-match odds, and write match-summary parquet files. |
| `01_generate_probabilities.ipynb` | Generate outcome and exact-score probabilities under five approaches: Poisson Binomial, Double Poisson, Bivariate Poisson, Dixon–Coles, Bet365 implied. |
| `02_evaluation.ipynb`             | Pool predictions across the five leagues, report Brier / RPS / reliability diagrams for outcomes (incl. odds baseline), empirical-vs-model goal/scoreline plots, TVD and KL divergence for exact scores. |
| `output/`                         | Generated `*.parquet` predictions (created by notebook 1). |
| `figures/`                        | Generated PNG figures (created by notebook 2). |
| `results/`                        | Main results of the analysis: Brier/RPS scores, TVD/KL divergences (CSV, created by notebook 2). |

## Results

### Outcome prediction accuracy (Brier score and RPS)

All xG-based approaches outperform pre-match betting odds on both metrics. 95 % bootstrap confidence intervals in brackets.

| Approach | N | Brier | [95 % CI] | RPS | [95 % CI] |
|---|---|---|---|---|---|
| Poisson Binomial | 1822 | 0.5391 | [0.520, 0.557] | 0.1768 | [0.170, 0.184] |
| Double Poisson   | 1822 | 0.5408 | [0.523, 0.558] | 0.1775 | [0.171, 0.184] |
| Bivariate Poisson | 1822 | 0.5405 | [0.524, 0.557] | 0.1774 | [0.171, 0.184] |
| Dixon–Coles      | 1822 | 0.5407 | [0.523, 0.558] | 0.1775 | [0.171, 0.184] |
| Betting Odds     | 1822 | 0.5871 | [0.573, 0.601] | 0.1992 | [0.193, 0.205] |

### Score distribution fit (TVD and KL divergence)

Pooled across all five leagues. Lower is better; Poisson Binomial produces the tightest fit to the empirical goal and scoreline distributions.

| Approach | TVD Home goals | TVD Away goals | TVD Total goals | TVD Scorelines | KL Home goals | KL Away goals | KL Total goals | KL Scorelines |
|---|---|---|---|---|---|---|---|---|
| Poisson Binomial  | 0.036 | 0.027 | 0.054 | 0.064 | 0.007 | 0.004 | 0.012 | 0.023 |
| Double Poisson    | 0.085 | 0.087 | 0.099 | 0.129 | 0.022 | 0.020 | 0.030 | 0.055 |
| Bivariate Poisson | 0.083 | 0.084 | 0.098 | 0.124 | 0.021 | 0.019 | 0.031 | 0.053 |
| Dixon–Coles       | 0.085 | 0.087 | 0.094 | 0.123 | 0.022 | 0.020 | 0.030 | 0.054 |

### Estimated model parameters

DIBP (Diagonal-Inflated Bivariate Poisson) and Dixon–Coles dependence parameters, fitted by MLE on observed scorelines per league. λ₃ captures covariance between team scores; p and θ govern draw inflation; ρ is the Dixon–Coles low-score adjustment.

| League | λ₃ (DIBP) | p (DIBP) | θ (DIBP) | ρ (Dixon–Coles) |
|---|---|---|---|---|
| Premier League | 0.0000 | 0.0558 | 1.3645 | −0.0775 |
| Bundesliga     | 0.0000 | 0.0001 | 1.4372 | −0.0647 |
| La Liga        | 0.0000 | 0.0145 | 1.4005 | −0.1003 |
| Serie A        | 0.0000 | 0.0113 | 1.7542 | +0.0318 |
| Ligue 1        | 0.0158 | 0.0121 | 1.4083 | −0.1027 |

## Reproducing the analysis

From this directory, with the project's Python environment active:

```bash
python -m nbconvert --to notebook --execute --inplace 00_data_acquisition.ipynb
python -m nbconvert --to notebook --execute --inplace 01_generate_probabilities.ipynb
python -m nbconvert --to notebook --execute --inplace 02_evaluation.ipynb
```

The notebooks reuse the analysis code in `../modules/` (`data_loading.py`,
`probability_generation.py`, `evaluation.py`).

### External inputs

Notebook 0 fetches shot events directly from StatsBomb's public API
(`statsbombpy`, no credentials required). The five football-data.co.uk
season files for 2015/16 must be placed under `../../data/odds/`:

| League         | Filename  | Source URL |
|----------------|-----------|------------|
| Premier League | `E0.csv`  | https://www.football-data.co.uk/mmz4281/1516/E0.csv  |
| Bundesliga     | `D1.csv`  | https://www.football-data.co.uk/mmz4281/1516/D1.csv  |
| La Liga        | `SP1.csv` | https://www.football-data.co.uk/mmz4281/1516/SP1.csv |
| Serie A        | `I1.csv`  | https://www.football-data.co.uk/mmz4281/1516/I1.csv  |
| Ligue 1        | `F1.csv`  | https://www.football-data.co.uk/mmz4281/1516/F1.csv  |

If a CSV is missing, notebook 0 still saves the league's parquet but without
odds, and notebook 1 will skip the betting-odds approach for that league.
