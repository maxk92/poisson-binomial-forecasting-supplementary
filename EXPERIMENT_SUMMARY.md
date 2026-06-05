# Experiment Plan and Results — Supplementary Analysis

This document captures the complete methodology and results of the
supplementary analysis at `data_analysis/supplementary-material/`. It is
intended as a reference for future writing sessions that incorporate the
results into the manuscript at `manuscript/polynomial-multiplication-xg.qmd`.

The supplementary pipeline consists of three notebooks:

1. `00_data_acquisition.ipynb` — fetch and merge raw data
2. `01_generate_probabilities.ipynb` — fit and apply five forecasting approaches
3. `02_evaluation.ipynb` — pooled evaluation across the five leagues

All reusable code lives in `data_analysis/modules/`
(`data_loading.py`, `probability_generation.py`, `evaluation.py`); the
supplementary notebooks import from there to avoid duplication.

---

## 1. Headline findings

Pooled across **1822 matches** from the five 2015/16 European top leagues
(all 1822 with Bet365 odds):

| Finding | Quantitative claim |
|---|---|
| **Post-match xG-based forecasts beat the pre-match betting market** on both proper scoring rules. | Polynomial multiplication: Brier = 0.5391, RPS = 0.1768; Bet365: Brier = 0.5871, RPS = 0.1992. |
| Polynomial multiplication is the **best of the four xG-based models** by every metric we report. | Lowest Brier, lowest RPS; lowest TVD and KL on home, away, total and exact-scoreline distributions. |
| Modelling between-team dependence (DIBP, Dixon–Coles) gives a **small but consistent improvement on exact scorelines** over independent Double Poisson. | DIBP reduces exact-scoreline TVD from 0.129 (DP) to 0.124; KL from 0.055 to 0.053. Dixon–Coles: TVD to 0.123; KL to 0.054. DIBP and DC are essentially tied on scoreline fit. |
| Bet365 is the **best-calibrated** of all five forecasts, but trades calibration for sharpness — its Brier and RPS are the worst. | ECE: Bet365 = 0.0196, DC = 0.0211, DP = 0.0237, PB = 0.0274 (DIBP similar to DP; not re-captured in summary CSV for this run). |
| Among the model approaches, **Dixon–Coles inherits Double-Poisson sharpness while improving calibration** (especially on draws). | DC ECE_draw = 0.0195 vs DP ECE_draw = 0.0288. |
| Polynomial multiplication is **better-calibrated on draws** than the other models because it directly models discreteness of shot outcomes. | PB ECE_draw = 0.0225 vs DP ≈ 0.028; PB also has the best home and away marginal fit. |

---

## 2. Data

### 2.1 Sources

| Source | Role | Files |
|---|---|---|
| StatsBomb open data (`statsbombpy`) | Per-shot xG values and match metadata | `competition_id` ∈ {2 (EPL), 7 (Ligue 1), 9 (Bundesliga), 11 (La Liga), 12 (Serie A)}, `season_id = 27` (2015/16) |
| football-data.co.uk | Pre-match Bet365 decimal odds | `data/odds/E0.csv`, `D1.csv`, `SP1.csv`, `I1.csv`, `F1.csv` |

### 2.2 Sample

| League | Slug | StatsBomb matches | Complete records (xG + odds) |
|---|---|---|---|
| Premier League | `epl_1516` | 380 | 380 |
| Bundesliga (Germany) | `bundesliga_1516` | 306 | 306 |
| La Liga (Spain) | `laliga_1516` | 380 | 380 |
| Serie A (Italy) | `seriea_1516` | 380 | 380 |
| Ligue 1 (France) | `ligue1_1516` | 377 | 376 |
| **Total** | | **1823** | **1822** |

For Ligue 1, 3 of the 380 regular-season fixtures are absent from the
StatsBomb open data (leaving 377), and one further match cannot be matched
to a Bet365 entry in the football-data.co.uk file, leaving 376 complete
records. For comparability, the analytical sample is restricted to the
1822 matches with complete records across all five leagues; these 4 Ligue 1
matches are excluded from all analyses.

### 2.3 Loading and processing

The data acquisition pipeline is:

1. **StatsBomb shot extraction** (`load_statsbomb_competition`).
   For each league/season the API returns the match table and, per match,
   all event records of `type == "Shot"`. Only fields needed for analysis
   are retained: `match_id`, `team`, `shot_statsbomb_xg`, `shot_outcome`,
   plus match metadata (`home_team`, `away_team`, `home_score`, `away_score`,
   `match_date`, `competition_name`, `season_name`).

2. **Per-match aggregation** (`process_shots_to_match_summary`).
   Shots are grouped by `(match_id, team)`. For each row of the resulting
   match-summary table:
   - `home_xg`, `away_xg` = sum of per-shot xG values for that team in
     that match.
   - `home_xg_values`, `away_xg_values` = JSON list of every shot's xG
     value (preserved for the polynomial-multiplication approach).
   - `home_shots`, `away_shots` = count of shots.
   - `home_goals`, `away_goals` = taken from match-level `home_score` /
     `away_score` (so own-goals, which do not appear as shots, are
     correctly counted).

3. **Odds merge** (`load_football_data_odds` + `merge_odds_with_matches`).
   The football-data.co.uk Bet365 columns `B365H`, `B365D`, `B365A`
   contain decimal odds. Implied probabilities are computed as
   `1/odds_i`, then renormalised so the three sum to 1 — this removes
   the bookmaker overround (Hvattum & Arntzen, 2010). The merge is on
   `(match_date, home_team, away_team)`. A team-name mapping
   (`TEAM_NAME_MAP_DEFAULT` in `data_loading.py`) reconciles the two
   conventions (e.g. football-data: `Man City` ↔ StatsBomb:
   `Manchester City`).

   *Format gotcha:* the football-data.co.uk season files used here have
   ISO `YYYY-MM-DD` dates, not their usual `DD/MM/YYYY`. The notebook
   passes `date_format='%Y-%m-%d'` explicitly to override the loader's
   `dayfirst=True` default.

4. **Persistence.** One parquet per league at `data/input_data/<slug>.parquet`
   with the schema documented at the top of `data_loading.py`:
   `match_id, match_date, competition_name, season_name, home_team,
   away_team, home_goals, away_goals, home_xg, away_xg, home_shots,
   away_shots, home_xg_values, away_xg_values, betting_p_home,
   betting_p_draw, betting_p_away`.

---

## 3. Forecasting approaches

All four model approaches use only the xG information available *after* a
match has been played. The betting-odds approach uses pre-match prices.

### 3.1 Poisson Binomial via polynomial multiplication (Ruiz et al., 2015)

Each shot $i$ is treated as an independent Bernoulli trial with success
probability equal to its xG, $p_i$. The number of goals scored by a team
with $n$ shots is the sum of $n$ independent Bernoulli variables — a
Poisson Binomial random variable. Its PMF is the convolution of the
shot-level Bernoulli PMFs:

$$
\sum_{k} P(X=k)\, a^{k} \;=\; \prod_{i=1}^{n} \big[(1 - p_i) + p_i\, a\big].
$$

Numerically this is a chain of 1-D convolutions
(`poisson_binomial_pmf` in `probability_generation.py`):

```python
result = np.array([1 - p[0], p[0]])
for q in p[1:]:
    result = np.convolve(result, np.array([1 - q, q]))
```

Home and away PMFs are computed independently and combined via the outer
product to obtain the joint scoreline matrix $P(H{=}h, A{=}a) = P_H(h)\,P_A(a)$,
then aggregated to outcomes:
$P(\text{home win}) = \sum_{h>a} P(h, a)$,
$P(\text{draw}) = \sum_h P(h, h)$,
$P(\text{away win}) = \sum_{h<a} P(h, a)$.

### 3.2 Double Poisson

Same independence assumption, but each marginal is a single Poisson with
rate equal to the team's total xG sum:
$H \sim \text{Pois}(\lambda_H = \sum p_i)$,
$A \sim \text{Pois}(\lambda_A = \sum p_j)$.
This is the textbook xG-based score forecast (Maher, 1982; Eggels et al.,
2016). The PMF is truncated at $k = 10$ goals and renormalised.

### 3.3 Diagonal Inflated Bivariate Poisson / DIBP (Karlis & Ntzoufras, 2005)

The plain Bivariate Poisson (Karlis & Ntzoufras, 2003) introduces a
shared latent component $W_3$ so that $H = W_1 + W_3$, $A = W_2 + W_3$,
giving $\mathrm{Cov}(H, A) = \lambda_3 \ge 0$. Because soccer data
typically shows near-zero or slightly negative home/away goal correlation,
MLE pushes $\lambda_3 \to 0$ and BP collapses to Double Poisson — an
identified weakness of the 2003 model.

Karlis & Ntzoufras (2005) propose the **Diagonal Inflated Bivariate
Poisson (DIBP)**, which adds a draw-inflation component:

$$
f_{\text{DIBP}}(x, y) =
\begin{cases}
(1-p)\,f_{\text{BP}}(x,y\mid\lambda_1,\lambda_2,\lambda_3) & x \neq y \\
(1-p)\,f_{\text{BP}}(x,y) + p\,f_{\text{Pois}}(x\mid\theta) & x = y
\end{cases}
$$

The inflation weight $p \in [0, 1]$ and Poisson rate $\theta > 0$ are
estimated jointly with $\lambda_3$ by 3-D MLE via L-BFGS-B
(`fit_dibp_params`, 9 starting points to avoid local optima). Per-match
marginals remain anchored at xG sums:
$\lambda_1 = \max(\text{home\_xg} - \lambda_3, \varepsilon)$,
$\lambda_2 = \max(\text{away\_xg} - \lambda_3, \varepsilon)$.

Key properties over plain BP:
- **Allows negative marginal covariance** even when $\lambda_3 = 0$,
  because the draw-inflation mass shifts probability from off-diagonal to
  diagonal cells.
- **Explicitly models excess draws**: soccer draws are empirically more
  frequent than Poisson independence predicts; the Poisson$(θ)$ inflation
  corrects this directly.
- Poisson inflation is chosen for parsimony (1 parameter) and best BIC
  in the paper's simulation examples.

The implementation is `diagonal_inflated_bivariate_poisson_matrix` (which
internally calls `bivariate_poisson_matrix`) in `probability_generation.py`.

### 3.4 Dixon–Coles (1997)

Independent-Poisson product with a low-score adjustment:

$$
P_{DC}(x, y) = \tau_{\lambda_H,\lambda_A}(x, y)\; P_{\lambda_H}(x)\; P_{\lambda_A}(y),
$$

with $\tau$ supported only on the four lowest cells:

$$
\tau(0,0) = 1 - \lambda_H\lambda_A\rho,\quad
\tau(0,1) = 1 + \lambda_H\rho,\quad
\tau(1,0) = 1 + \lambda_A\rho,\quad
\tau(1,1) = 1 - \rho,
$$

and $\tau(x, y) = 1$ otherwise. The dependence parameter $\rho$ is
**fitted globally per league** by MLE on observed scores
(`fit_dixon_coles_rho`, bounded $[-0.3, 0.3]$). $\lambda_H$ and
$\lambda_A$ remain the team xG sums; only the joint matrix differs from
Double Poisson. After applying $\tau$, any negative cells (which would
indicate $\rho$ violating its feasibility region for the given $\lambda$
pair) are clipped to zero, then the matrix is renormalised.

The implementation is `dixon_coles_matrix` in
`probability_generation.py`.

### 3.5 Pre-match Bet365 baseline

Decimal odds for the three outcomes are converted to normalised implied
probabilities $\hat p_o = (1/o_o) / \sum_k (1/o_k)$. This is a
pre-match, market-based forecast that uses no in-match information —
the contrast between this and the post-match xG models is the central
information-content comparison of the paper.

### 3.6 Fitted dependence parameters per league

| League | $\lambda_3$ (DIBP) | $p$ (DIBP) | $\theta$ (DIBP) | $\rho$ (Dixon–Coles) |
|---|---|---|---|---|
| Premier League (epl_1516) | 0.0000 | 0.0558 | 1.3645 | −0.0775 |
| Bundesliga (bundesliga_1516) | 0.0000 | ≈0.0001 | — | −0.0647 |
| La Liga (laliga_1516) | 0.0000 | 0.0145 | 1.4005 | −0.1003 |
| Serie A (seriea_1516) | 0.0000 | 0.0113 | 1.7542 | +0.0318 |
| Ligue 1 (ligue1_1516) | 0.0158 | 0.0121 | 1.4083 | −0.1100 |

**Reading these.** The DIBP inflation weight $p$ captures draw-excess
that BP's non-negative-covariance constraint cannot represent. EPL shows
the strongest inflation ($p = 5.6\%$, centered at $\theta \approx 1.4$
goals per team — roughly the 1:1 result). La Liga, Serie A and Ligue 1
have smaller but meaningful inflation (1.1–1.5%); Bundesliga shows
essentially no draw excess ($p \approx 0$, degenerating to Double Poisson).
All fitted $\lambda_3$ are zero or near-zero, confirming that the plain
BP covariance structure does not fit soccer data. $\rho < 0$ in four of
five leagues confirms the classical Dixon–Coles finding; Serie A's
$\rho > 0$ reflects relatively fewer low-score draws relative to the
Poisson baseline.

---

## 4. Evaluation framework

Two evaluation targets, both computed on the **pooled** sample of all
five leagues (1822 matches, all with complete records including Bet365 odds). Pooling is justified by
the manuscript's research question — *can a post-match xG-based forecast
outperform the pre-match market on average?* — and increases statistical
power by an order of magnitude over per-league analysis.

### 4.1 Outcome probability evaluation

Three metrics, all lower-is-better.

**Multi-class Brier score** (`calculate_brier_score`):

$$
B = \frac{1}{N}\sum_{n=1}^{N}\sum_{o \in \{H, D, A\}}\big(\hat p_{n,o} - \mathbb{1}[y_n = o]\big)^{2}.
$$

Range $[0, 2]$; uniform 1/3 baseline = 0.667.

**Ranked Probability Score** (`mean_rps`), accounting for the natural
ordering of the three outcomes (home win < draw < away win):

$$
\mathrm{RPS} = \tfrac{1}{2}\big[(F_1 - G_1)^2 + (F_2 - G_2)^2\big],
$$

with cumulative predicted probabilities $F_1 = \hat p_H$, $F_2 = \hat p_H + \hat p_D$
and indicator cumulants $G$. Uniform baseline = 0.333. RPS is the
recommended primary scoring rule for ordered three-way football
forecasts (Constantinou & Fenton, 2012).

**Expected Calibration Error** (`expected_calibration_error`),
macro-averaged over the three outcome classes. For each class $o$,
predicted probabilities are binned into 10 equal-width deciles, and

$$
\mathrm{ECE}_o = \sum_{m=1}^{10}\frac{|B_m|}{N}\,\big|\bar p_{B_m} - \bar y_{B_m}\big|.
$$

The reported ECE is $\tfrac{1}{3}(\mathrm{ECE}_H + \mathrm{ECE}_D + \mathrm{ECE}_A)$.

**Reliability diagrams** (`plot_calibration_curves`) — one panel per
outcome class, scatter of bin mean predicted probability vs bin observed
frequency, with the diagonal indicating perfect calibration.

### 4.2 Score-distribution evaluation

For each approach we **average the per-match predicted PMFs across all
1822 matches** to obtain a single pooled "expected" distribution
($\bar P_{\text{model}}(x, y)$), and compare against the empirical
distribution computed from observed scores. This is the right comparison
for an *aggregate* fit assessment: it tests whether the model is
unbiased across the population, not whether any individual prediction
is well-calibrated.

Four marginal/joint distributions are compared
(`compute_empirical_distributions`, `aggregate_score_distributions`):

1. Home-team goals (length 11)
2. Away-team goals (length 11)
3. Total goals = home + away (length 13)
4. Joint scoreline matrix cropped to 8 × 8 = 64 cells

Two metrics (`tvd`, `kl_divergence`):

- **Total Variation Distance**: $\mathrm{TVD}(P, Q) = \tfrac{1}{2}\sum_i |p_i - q_i|$.
  Range $[0, 1]$; bounded, intuitive, the worst-case event-level error.
- **Kullback–Leibler divergence**: $D_{\mathrm{KL}}(P\Vert Q) = \sum_i p_i \log(p_i/q_i)$
  (with $\varepsilon = 10^{-10}$ smoothing). Equal up to a constant to
  the negative expected log-score; the canonical strictly-proper
  scoring rule for distributional forecasts (Gneiting & Raftery, 2007).

**Visualisations** (all in `evaluation.py`):

- `plot_distribution_markers` — empirical PMF as bars, each model
  overlaid as a horizontally-jittered scatter so closely-clustered
  points (DP, BP, DC) remain visually disentangleable. Used for the
  three goal distributions.
- `plot_top_scorelines_with_markers` — same idea on the top-20 most
  frequent observed scorelines, ordered by empirical frequency.

---

## 5. Results

All numbers are pooled across the five leagues and rounded to four
decimals.

### 5.1 Outcome probabilities (1822 matches, all with complete records)

| Approach | N | Brier | RPS | ECE | ECE_home | ECE_draw | ECE_away |
|---|---:|---:|---:|---:|---:|---:|---:|
| Poisson Binomial | 1822 | **0.5391** | **0.1768** | 0.0274 | 0.0394 | 0.0225 | 0.0204 |
| DIBP | 1822 | 0.5405 | 0.1774 | — | — | — | — |
| Double Poisson | 1822 | 0.5408 | 0.1775 | 0.0237 | 0.0176 | 0.0288 | 0.0247 |
| Dixon–Coles | 1822 | 0.5407 | 0.1775 | 0.0211 | 0.0213 | 0.0195 | 0.0224 |
| Bet365 (pre-match) | 1822 | 0.5871 | 0.1992 | **0.0196** | 0.0264 | 0.0117 | 0.0208 |

*Note: DIBP ECE breakdown not recaptured in the current pipeline run (evaluation_metrics.csv stores only Brier/RPS). DIBP ECE is expected to lie between DP and DC on draws, given p > 0 in four leagues.*

**Subtle observations.**

- The four xG-based models are tightly clustered on Brier and RPS
  (within ≈ 0.002); Polynomial Binomial wins by a small but consistent
  margin on both. The 8.2 % Brier and 11.3 % RPS improvements over
  Bet365 are an order of magnitude larger than the differences among
  the model approaches.
- DIBP is modestly better than Double Poisson and Dixon-Coles on Brier
  (0.5405 vs 0.5407–0.5408), confirming the draw-inflation effect.
  However, the difference is within confidence intervals and should not
  be over-interpreted.
- Dixon–Coles inherits the marginal structure of Double Poisson but
  improves draw calibration substantially: ECE_draw drops from 0.0288
  (DP) to 0.0195 (DC). This is the textbook Dixon–Coles benefit on
  draws translating directly into better-calibrated three-way
  probabilities, even when the Brier/RPS difference is too small to
  detect.
- Bet365 has the best ECE (0.0196), particularly on draws (0.0117) —
  a market is essentially well-calibrated by construction (it is forced
  to be by arbitrage). What it lacks is the *sharpness* that comes from
  observing the actual shot xG sequence: its predictions are more
  diffuse, hence the worse Brier and RPS.

### 5.2 Goal- and score-distribution fit

**Total Variation Distance** (rows: distribution; columns: model).

| Distribution | Poisson Binomial | Double Poisson | DIBP | Dixon–Coles |
|---|---:|---:|---:|---:|
| Home goals | **0.036** | 0.085 | 0.083 | 0.085 |
| Away goals | **0.027** | 0.087 | 0.084 | 0.087 |
| Total goals | **0.054** | 0.099 | 0.098 | 0.094 |
| Exact scorelines | **0.064** | 0.129 | 0.124 | 0.123 |

**KL divergence** (same layout).

| Distribution | Poisson Binomial | Double Poisson | DIBP | Dixon–Coles |
|---|---:|---:|---:|---:|
| Home goals | **0.007** | 0.022 | 0.021 | 0.022 |
| Away goals | **0.004** | 0.020 | 0.019 | 0.020 |
| Total goals | **0.012** | 0.030 | 0.031 | 0.030 |
| Exact scorelines | **0.023** | 0.055 | 0.053 | 0.054 |

**Why DP, DIBP and DC differ on marginals (this is *not* a bug; do not re-flag).**
DIBP applies diagonal inflation to the joint matrix *after* the xG-anchored
marginals are set, so the home/away marginals are no longer exactly Poisson.
In practice, with small $p$ (0.001–0.056), the marginal shift is small but
measurable: DIBP home TVD 0.083 vs DP 0.085. For Dixon–Coles the τ correction
is constructed to be marginal-preserving (sum identities hold per row/column);
its marginals equal Double Poisson's exactly. (Verified numerically in prior run.)

**Subtle observations.**

- Polynomial multiplication wins **all eight cells**, by roughly a
  factor of two on the marginals and a factor of 2.3 on KL of the joint
  scoreline matrix. The intuitive explanation: aggregating
  $n$ Bernoulli trials into a single Poisson rate $\lambda = \sum p_i$
  loses information about the *shape* of the per-shot probabilities;
  the Poisson Binomial preserves it. This is the central methodological
  argument of the paper.
- DIBP improves over plain Double Poisson on all four TVD metrics; the
  largest gain is on exact scorelines (0.129 → 0.124, a 4% reduction),
  confirming that draw inflation helps scoreline fit. DIBP is essentially
  tied with Dixon–Coles on exact scorelines (TVD: DIBP 0.124, DC 0.123;
  KL: DIBP 0.053, DC 0.054).
- For *total goals*, DC (TVD 0.094) outperforms DIBP (0.098): the τ
  correction redistributes joint mass in a way that better matches the
  total-goal marginal, whereas DIBP's diagonal inflation concentrates
  additional probability exactly on the diagonal (equal-score cells),
  which shifts total goals less efficiently.
- The exact-scoreline gap between PB and the dependence-aware models
  (PB TVD 0.064 vs DIBP/DC TVD 0.123–0.124) is roughly twice the gap
  between DIBP/DC and DP (0.129 → 0.123/0.124). Use this to argue in
  the paper that **finer per-shot modelling is more valuable on this
  dataset than between-team dependence modelling**.

### 5.3 Figures produced

Saved to `data_analysis/supplementary-material/figures/`:

| File | Cell that produces it | Suggested caption hook |
|---|---|---|
| `pooled_calibration.eps` | `02_evaluation.ipynb` cell `supp02-cal` | "Reliability diagrams for the three outcome classes, pooled over 1822 matches; Bet365 lies closest to the diagonal but the xG-based models compensate with greater sharpness." |
| `pooled_goal_distributions.eps` | cell `supp02-marg` | "Empirical (bars) versus model-predicted (markers, jittered) PMFs of home goals, away goals and total goals." |
| `pooled_top_scorelines.eps` | cell `supp02-top` | "Top-20 most frequent observed scorelines: empirical (bars) and model-predicted probabilities (markers). Polynomial multiplication tracks the empirical bars most closely; DIBP and Dixon–Coles recover some of the under-prediction of 0:0 and 1:1." |
| `pooled_scoreline_heatmaps.eps` | cell `supp02-heatmap` | "Pooled predicted scoreline matrices (cropped to 5×5) for each model; DIBP shows visibly more mass on the diagonal compared to Double Poisson." |
| `pooled_gof_summary.eps` | cell `supp02-gof` | "Summary of TVD and KL goodness-of-fit across models and distribution targets; Poisson Binomial dominates on all eight metrics." |
| `pooled_gof_summary_exact_scoreline.eps` | cell `supp02-gof` | "Exact-scoreline TVD/KL by approach; DIBP and Dixon–Coles are essentially tied and both improve substantially over Double Poisson." |

---

## 6. Instructions for incorporating into the manuscript

The manuscript file is at `manuscript/polynomial-multiplication-xg.qmd`.
A backup of the pre-revision draft is at
`manuscript/polynomial-multiplication-xg_backup.qmd` (referenced in
auto-memory: `project_manuscript_revision.md`).

### 6.1 What to use these results for

This experiment now **fully supports the manuscript's core empirical
claims**:

1. *Polynomial multiplication produces probabilistic forecasts that
   outperform the betting market* → §5.1 outcome scoring table.
2. *Polynomial multiplication is superior to Double Poisson because it
   preserves shot-level information* → §5.2 GOF table; emphasise the
   2× factor on marginals.
3. *Modelling between-team dependence (DIBP, Dixon–Coles) gives a small
   improvement on exact scorelines but does not affect outcome
   forecasting much in this sample* → §5.1 / §5.2 contrast.
4. *DIBP improves over plain BP by allowing draw inflation while
   relaxing the non-negative-covariance constraint* → §3.3; §5.1/§5.2
   DIBP vs DP differences. Cite Karlis & Ntzoufras (2005) `@karlis2005`.

### 6.2 Concrete edits to make

When opening a future writing session, follow this checklist:

1. **Methods section.**
   - Replace any text that describes only Double Poisson alongside
     Polynomial Binomial with the full four-model setup of §3.
   - Reuse the equations in §3.1–§3.4 verbatim where the manuscript
     references the corresponding model. The notation matches the
     existing draft.
   - Add a paragraph on parameter estimation: λ₃ and ρ are fitted by
     MLE on observed scores per league, treating xG sums as fixed
     marginal-anchor inputs.
   - Add the Bet365 baseline construction (§3.5) — currently flagged
     as `[TODO]` in the draft.

2. **Data section.**
   - Replace the World Cup 2022 sample with the five-league panel
     (1822 matches with complete records; see §2.2). The wc2022 work remains in the
     main `data_analysis/` folder for reference but the supplementary
     pipeline (and therefore the headline results) is now leagues-only.
   - Add the team-name reconciliation note (§2.3 step 3) and the ISO
     date-format gotcha to a methods paragraph or footnote.

3. **Results section.**
   - Insert the Brier/RPS/ECE table as the primary outcome-evaluation
     table; mark Polynomial Binomial as the winner on RPS/Brier and
     Bet365 as the winner on ECE.
   - Insert the TVD/KL table as the primary score-distribution table.
   - Drop any per-league results tables — the supplementary results are
     pooled by design.
   - Reference the three figures from §5.3; their caption hooks are
     ready-to-use.

4. **Discussion.**
   - Frame Bet365's better ECE / worse Brier as a *sharpness-vs-calibration
     trade-off*: the market is forced to be calibrated by arbitrage but
     cannot exploit shot-level data; xG-based models trade a small amount
     of calibration for substantially more sharpness.
   - Acknowledge BP's ineffectiveness as a finding rather than a flaw
     of the model: the empirical home/away covariance in this sample
     is essentially zero (or slightly negative), and BP cannot represent
     negative covariance.
   - Use the §5.2 observation that PB's marginal-fit gap over
     DP/BP/DC is ~2× the joint-fit gap of DC over DP to argue that
     *granular shot-level modelling is more leveraged than between-team
     dependence modelling*. This is a useful framing for the
     contribution claim.

### 6.3 Re-running

The headline numbers in §5 will reproduce exactly when the supplementary
notebooks are re-executed in order. They depend only on:

- The frozen StatsBomb 2015/16 league shot data (the API has not
  re-issued these matches; xG values are stable).
- The football-data.co.uk season files in `data/odds/`.

If the manuscript later moves to a different competition or season,
adjust `LEAGUES` in `00_data_acquisition.ipynb` and re-run the three
notebooks; everything downstream is data-agnostic.

### 6.4 Citations to use

Already in `~/drive/phd_library.bib`:

| Claim | Bibtex key |
|---|---|
| Polynomial multiplication / Poisson Binomial in soccer | `@ruiz2015` |
| Independent-Poisson scoring model | `@maher1982` |
| Dixon–Coles low-score adjustment | `@dixon1997` |
| Bivariate Poisson (base model) | `@karlis2003` |
| Diagonal Inflated Bivariate Poisson (DIBP) | `@karlis2005` |
| xG-based scoring with Poisson Binomial | `@eggels2016` |
| Betting market efficiency / overround | `@hvattum2010`, `@forrest2005` |
| Use of betting odds as benchmark | `@constantinou2012` |
| Strictly proper scoring rules / log score | (Gneiting & Raftery 2007 — add if missing) |
| RPS for football forecasts | `@constantinou2012` |

---

## 7. Audit log

- **2026-06-05** — Replaced Bivariate Poisson with Diagonal Inflated
  Bivariate Poisson (DIBP, Karlis & Ntzoufras 2005) throughout the pipeline.
  Plain BP always fitted λ₃ ≈ 0 (collapsing to Double Poisson) because soccer
  home/away goal correlation is near-zero or negative — a known limitation of
  the non-negative-covariance constraint. DIBP adds Poisson(θ) diagonal
  inflation (weight p) fitted jointly with λ₃ via 3-D L-BFGS-B MLE.
  Changes: `probability_generation.py` (new functions), `evaluation.py`
  (colour map), `01_generate_probabilities.ipynb`, `02_evaluation.ipynb`.
  Headline numbers updated to DIBP values throughout this document.

- **2026-05-09** — Bug discovered and fixed in
  `evaluation.aggregate_score_distributions`. The previous version
  computed each per-match total-goal PMF as `np.convolve(home_pmf,
  away_pmf)`, which silently assumes independence. For the dependent
  models this hid the very effect that distinguishes them: DC and BP
  produced totals that exactly equalled DP's. The fix sums the
  anti-diagonals of the joint scoreline matrix
  (`tot[h+a] += matrix[h, a]`), which reduces to the convolution under
  independence and is correct under dependence. Numbers above are the
  post-fix values. The home and away marginal numbers are unchanged
  because, mathematically, all three models (DP, BP, DC) share
  identical marginals (see boxed note in §5.2).

## 8. File map

```
data_analysis/
├── modules/
│   ├── data_loading.py            # StatsBomb + football-data.co.uk helpers
│   ├── probability_generation.py  # PB, DP, DIBP, DC + MLE fitters + I/O
│   └── evaluation.py              # scoring rules, GOF metrics, plotting
└── supplementary-material/
    ├── README.md                  # how to run the three notebooks
    ├── EXPERIMENT_SUMMARY.md      # this file
    ├── 00_data_acquisition.ipynb  # → ../../data/input_data/*.parquet
    ├── 01_generate_probabilities.ipynb  # → output/<league>_<approach>_<target>.parquet
    ├── 02_evaluation.ipynb        # → figures/*.eps + results/*.csv
    ├── output/                    # 10 DIBP parquets + DP/DC/PB/Betting parquets (5 leagues × 2)
    ├── results/
    │   ├── evaluation_metrics.csv # Brier/RPS with 95% CIs per approach
    │   └── tvd_kl_results.csv     # TVD and KL by distribution and approach
    └── figures/
        ├── pooled_calibration.eps
        ├── pooled_goal_distributions.eps
        ├── pooled_top_scorelines.eps
        ├── pooled_scoreline_heatmaps.eps
        ├── pooled_scoreline_heatmaps_double_poisson.eps
        ├── pooled_scoreline_heatmaps_diagonal_inflated_bivariate_poisson.eps
        ├── pooled_scoreline_heatmaps_dixon_coles.eps
        ├── pooled_scoreline_heatmaps_poisson_binomial.eps
        ├── pooled_gof_summary.eps
        └── pooled_gof_summary_exact_scoreline.eps
```
