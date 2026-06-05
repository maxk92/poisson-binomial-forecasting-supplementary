# Supplementary Material

Three-notebook companion to *Polynomial Multiplication for Post-Match Soccer
Outcome Probabilities*, reproducing the full pipeline end-to-end on the five
2015/16 European top leagues. The analytical sample comprises **1822 matches**
with complete records: both StatsBomb shot-level xG data and Bet365 pre-match
odds. StatsBomb provides 377 of the 380 Ligue 1 matches (3 are absent from
the open data), and one of those 377 cannot be matched to a Bet365 odds entry,
leaving 376 complete Ligue 1 records. The remaining four leagues have full
coverage (380 + 306 + 380 + 380 = 1446 matches). Notebook 0 enforces this
filter at save time.

## Contents

| File | Purpose |
|---|---|
| `00_data_acquisition.ipynb`       | Download StatsBomb shot events for the five leagues, merge in football-data.co.uk pre-match odds, and write match-summary parquet files. |
| `01_generate_probabilities.ipynb` | Generate outcome and exact-score probabilities under five approaches: Poisson Binomial, Double Poisson, Bivariate Poisson, Dixon–Coles, Bet365 implied. |
| `02_evaluation.ipynb`             | Pool predictions across the five leagues, report Brier / RPS / reliability diagrams for outcomes (incl. odds baseline), empirical-vs-model goal/scoreline plots, TVD and KL divergence for exact scores. |
| `output/`                         | Generated `*.parquet` predictions (created by notebook 1). |
| `figures/`                        | Generated PNG figures (created by notebook 2). |

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
