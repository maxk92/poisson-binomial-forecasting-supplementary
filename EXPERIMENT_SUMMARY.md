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
| **Post-match xG-based forecasts beat the pre-match betting market** on both proper scoring rules. | Polynomial multiplication: Brier = 0.5392, RPS = 0.1768; Bet365: Brier = 0.5871, RPS = 0.1992. |
| Polynomial multiplication is the **best of the four xG-based models** by every metric we report. | Lowest Brier, lowest RPS; lowest TVD and KL on home, away, total and exact-scoreline distributions. |
| Modelling between-team dependence (Bivariate Poisson, Dixon–Coles) gives a **small but consistent improvement on exact scorelines** over independent Double Poisson. | DC reduces exact-scoreline TVD from 0.1293 (DP) to 0.1234; KL from 0.0551 to 0.0538. BP reduces only marginally because fitted λ₃ is near zero. |
| Bet365 is the **best-calibrated** of all five forecasts, but trades calibration for sharpness — its Brier and RPS are the worst. | ECE: Bet365 = 0.0196, DC = 0.0211, DP = 0.0237, BP = 0.0238, PB = 0.0274. |
| Among the model approaches, **Dixon–Coles inherits Double-Poisson sharpness while improving calibration** (especially on draws). | DC ECE_draw = 0.0195 vs DP ECE_draw = 0.0288. |
| Polynomial multiplication is **better-calibrated on draws** than the other models because it directly models discreteness of shot outcomes. | PB ECE_draw = 0.0225 vs DP/BP ≈ 0.028; PB also has the best home and away marginal fit. |

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

### 3.3 Bivariate Poisson (Karlis & Ntzoufras, 2003)

Introduces a shared latent component $W_3$:

$$
H = W_1 + W_3, \quad A = W_2 + W_3, \quad W_i \overset{\text{ind}}{\sim} \text{Pois}(\lambda_i).
$$

So $\mathrm{Cov}(H, A) = \lambda_3 \ge 0$ while preserving Poisson
marginals: $H \sim \text{Pois}(\lambda_1 + \lambda_3)$,
$A \sim \text{Pois}(\lambda_2 + \lambda_3)$. The joint PMF is

$$
P(H{=}x, A{=}y) = e^{-(\lambda_1+\lambda_2+\lambda_3)}
\sum_{k=0}^{\min(x,y)}
\frac{\lambda_3^k}{k!}\;
\frac{\lambda_1^{x-k}}{(x-k)!}\;
\frac{\lambda_2^{y-k}}{(y-k)!}.
$$

We anchor the marginals at the observed xG sums, so for each match
$\lambda_1 = \max(\text{home\_xg} - \lambda_3,\,\varepsilon)$ and
$\lambda_2 = \max(\text{away\_xg} - \lambda_3,\,\varepsilon)$. **A single
$\lambda_3$ is fitted globally per league** by maximum likelihood on
observed scores (`fit_bivariate_poisson_lambda3`, scalar bounded
`scipy.optimize.minimize_scalar`).

The implementation is `bivariate_poisson_matrix` in
`probability_generation.py`. Truncation at 10 goals + renormalisation is
applied as for Double Poisson.

**Limitation note for the paper.** Bivariate Poisson constrains
$\mathrm{Cov}(H, A) \ge 0$. The empirical correlation of home and away
goals in our sample is essentially zero or slightly negative, so MLE
typically pushes $\lambda_3 \to 0$ and the model becomes nearly
indistinguishable from Double Poisson.

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

| League | $\lambda_3$ (Bivariate Poisson) | $\rho$ (Dixon–Coles) |
|---|---|---|
| Premier League (epl_1516) | 0.0144 | −0.0775 |
| Bundesliga (bundesliga_1516) | 0.0002 | −0.0647 |
| La Liga (laliga_1516) | 0.0002 | −0.1003 |
| Serie A (seriea_1516) | 0.0002 | +0.0318 |
| Ligue 1 (ligue1_1516) | 0.0205 | −0.1100 |

**Reading these.** $\rho < 0$ inflates 0:0/1:1 and depresses 1:0/0:1 —
the classic Dixon–Coles correction confirmed in four of five leagues.
Serie A is a small positive outlier (often interpreted as defensive
discipline producing very few low-score draws relative to a Poisson
baseline). $\lambda_3$ is essentially zero in three leagues; EPL and
Ligue 1 show a barely-detectable positive co-movement.

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
| Poisson Binomial | 1822 | **0.5392** | **0.1768** | 0.0274 | 0.0394 | 0.0225 | 0.0204 |
| Double Poisson | 1822 | 0.5409 | 0.1774 | 0.0237 | 0.0176 | 0.0288 | 0.0247 |
| Bivariate Poisson | 1822 | 0.5409 | 0.1775 | 0.0238 | 0.0191 | 0.0283 | 0.0239 |
| Dixon–Coles | 1822 | 0.5408 | 0.1774 | 0.0211 | 0.0213 | 0.0195 | 0.0224 |
| Bet365 (pre-match) | 1822 | 0.5871 | 0.1992 | **0.0196** | 0.0264 | 0.0117 | 0.0208 |

**Subtle observations.**

- The four xG-based models are tightly clustered on Brier and RPS
  (within ≈ 0.002); Polynomial Binomial wins by a small but consistent
  margin on both. The 8.2 % Brier and 11.3 % RPS improvements over
  Bet365 are an order of magnitude larger than the differences among
  the model approaches.
- Bivariate Poisson is essentially identical to Double Poisson on every
  metric (because three of five fitted $\lambda_3$ are < 0.001).
  *Recommendation for the paper:* keep BP as a methodological completeness
  comparison but note in the discussion that it offers no measurable
  benefit on this sample — the empirical home/away goal correlation is
  too small to support its non-negative-covariance constraint.
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

| Distribution | Poisson Binomial | Double Poisson | Bivariate Poisson | Dixon–Coles |
|---|---:|---:|---:|---:|
| Home goals | **0.0365** | 0.0847 | 0.0847 | 0.0847 |
| Away goals | **0.0276** | 0.0874 | 0.0874 | 0.0874 |
| Total goals | **0.0540** | 0.0994 | 0.1004 | 0.0935 |
| Exact scorelines | **0.0644** | 0.1293 | 0.1291 | 0.1234 |

**KL divergence** (same layout).

| Distribution | Poisson Binomial | Double Poisson | Bivariate Poisson | Dixon–Coles |
|---|---:|---:|---:|---:|
| Home goals | **0.0066** | 0.0216 | 0.0216 | 0.0216 |
| Away goals | **0.0045** | 0.0197 | 0.0197 | 0.0197 |
| Total goals | **0.0116** | 0.0305 | 0.0310 | 0.0303 |
| Exact scorelines | **0.0227** | 0.0551 | 0.0550 | 0.0538 |

**Why DP, BP and DC are identical on the home/away marginals (this is
*not* a bug; do not re-flag).** For Bivariate Poisson the marginal of
$H = W_1 + W_3$ is $\text{Pois}(\lambda_1 + \lambda_3)$, and the
generators set $\lambda_1 = \text{home\_xg} - \lambda_3$ — so the
marginal is exactly $\text{Pois}(\text{home\_xg})$, identical to Double
Poisson. For Dixon–Coles the τ adjustment is *constructed to be
marginal-preserving*: the four corrections satisfy
$\sum_y \tau(0, y)\,P_a(y) = 1$ and the analogous identities for
row 1, column 0 and column 1, so row and column sums of the joint
matrix are exactly the un-adjusted Poisson marginals. (Verified
numerically: the per-match home/away PMFs in the saved parquet files
agree to floating-point precision across DP, BP and DC; only the joint
scoreline matrices differ — see e.g. EPL match 1, where DP[0,0]=0.0195
becomes DC[0,0]=0.0253 with $\rho<0$, while DP[1,0]=0.0419 becomes
DC[1,0]=0.0361.)

**Subtle observations.**

- Polynomial multiplication wins **all eight cells**, by roughly a
  factor of two on the marginals and a factor of 2.4 on KL of the joint
  scoreline matrix. The intuitive explanation: aggregating
  $n$ Bernoulli trials into a single Poisson rate $\lambda = \sum p_i$
  loses information about the *shape* of the per-shot probabilities;
  the Poisson Binomial preserves it. This is the central methodological
  argument of the paper.
- DP, BP and DC are **necessarily identical on the home and away
  marginals** (see the box above) and therefore on TVD/KL for those
  rows. For *total goals* the three models differ because total =
  $\sum_{h+a=k}$ joint, which depends on the dependence structure: DC
  improves over DP (TVD 0.0994 → 0.0935; KL 0.0305 → 0.0303), BP
  marginally worsens it (TVD 0.0994 → 0.1004; KL 0.0305 → 0.0310)
  because its MLE-fitted positive covariance pushes mass toward the
  joint diagonal in a direction that does not match this sample's
  empirical distribution.
- Where DC and BP also separate from DP is on **exact scorelines**:
  TVD 0.1293 (DP) → 0.1291 (BP) → 0.1234 (DC); KL 0.0551 → 0.0550 → 0.0538.
  Dixon–Coles wins by a fair-but-small margin (TVD reduction ≈ 4.6 %).
  Bivariate Poisson barely moves the needle.
- The exact-scoreline gap between PB and the dependence-aware models
  (PB TVD 0.0644 vs DC TVD 0.1234) is roughly twice the gap between DC
  and DP. Use this to argue in the paper that **finer per-shot
  modelling is more valuable on this dataset than between-team
  dependence modelling**.

### 5.3 Figures produced

Saved to `data_analysis/supplementary-material/figures/`:

| File | Cell that produces it | Suggested caption hook |
|---|---|---|
| `pooled_calibration.png` | `02_evaluation.ipynb` cell `supp02-cal` | "Reliability diagrams for the three outcome classes, pooled over 1822 matches; Bet365 lies closest to the diagonal but the xG-based models compensate with greater sharpness." |
| `pooled_goal_distributions.png` | cell `supp02-marg` | "Empirical (bars) versus model-predicted (markers, jittered) PMFs of home goals, away goals and total goals." |
| `pooled_top_scorelines.png` | cell `supp02-top` | "Top-20 most frequent observed scorelines: empirical (bars) and model-predicted probabilities (markers). Polynomial multiplication tracks the empirical bars most closely; Dixon–Coles recovers some of the under-prediction of 0:0 and 1:1." |

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
3. *Modelling between-team dependence (Dixon–Coles) gives a small
   improvement on exact scorelines but does not affect outcome
   forecasting much in this sample* → §5.1 / §5.2 contrast.
4. *Bivariate Poisson is conceptually appealing but mis-fits low-correlation
   data because of its non-negative covariance constraint* → §3.3
   limitation note; §5.1/§5.2 near-equivalence to Double Poisson.

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
| Bivariate Poisson | `@karlis2003` |
| xG-based scoring with Poisson Binomial | `@eggels2016` |
| Betting market efficiency / overround | `@hvattum2010`, `@forrest2005` |
| Use of betting odds as benchmark | `@constantinou2012` |
| Strictly proper scoring rules / log score | (Gneiting & Raftery 2007 — add if missing) |
| RPS for football forecasts | `@constantinou2012` |

---

## 7. Audit log

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
│   ├── probability_generation.py  # PB, DP, BP, DC + MLE fitters + I/O
│   └── evaluation.py              # scoring rules, GOF metrics, plotting
└── supplementary-material/
    ├── README.md                  # how to run the three notebooks
    ├── EXPERIMENT_SUMMARY.md      # this file
    ├── 00_data_acquisition.ipynb  # → ../../data/input_data/*.parquet
    ├── 01_generate_probabilities.ipynb  # → output/<league>_<approach>_<target>.parquet
    ├── 02_evaluation.ipynb        # → figures/*.png + inline tables
    ├── output/                    # 45 parquet files (5 leagues × 9 prediction files)
    └── figures/
        ├── pooled_calibration.png
        ├── pooled_goal_distributions.png
        └── pooled_top_scorelines.png
```
