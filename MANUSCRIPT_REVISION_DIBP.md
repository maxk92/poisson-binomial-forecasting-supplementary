# Manuscript Revision Instructions — Bivariate Poisson → DIBP

**Date:** 2026-06-05  
**Manuscript:** `manuscript/polynomial-multiplication-xg.qmd`  
**Purpose:** The Bivariate Poisson approach in the analysis pipeline was
upgraded from plain BP (Karlis & Ntzoufras, 2003) to the Diagonal Inflated
Bivariate Poisson (DIBP, Karlis & Ntzoufras, 2005). The pipeline still
labels the approach "Bivariate Poisson" in all figures and tables so that
reader-facing output does not change. This document specifies every place
in the manuscript that needs to be updated to reflect the new model,
its parameters, and the changed results.

---

## 1. What changed and why (context for the writing session)

### 1.1 The old model: plain Bivariate Poisson (Karlis & Ntzoufras, 2003)

Plain BP introduces a shared Poisson component W₃ so that
H = W₁ + W₃ and A = W₂ + W₃, giving Cov(H,A) = λ₃ ≥ 0.
The single scalar λ₃ was fitted per league by MLE.

**Problem:** Soccer home/away goal correlation is near-zero or slightly
negative. MLE pushed λ₃ → 0 in three of five leagues, making BP
numerically identical to Double Poisson. The "Bivariate Poisson" row in
all tables therefore showed the same Brier/RPS/TVD/KL as Double Poisson.

### 1.2 The new model: DIBP (Karlis & Ntzoufras, 2005)

DIBP adds a Poisson(θ) diagonal inflation:

    f_DIBP(x,y) = (1−p)·f_BP(x,y | λ₁,λ₂,λ₃)               for x ≠ y
    f_DIBP(x,y) = (1−p)·f_BP(x,y | λ₁,λ₂,λ₃) + p·Pois(x|θ)  for x = y

Parameters (λ₃, p, θ) are fitted jointly per league by 3-D L-BFGS-B MLE
(9 starting points). Per-match marginals remain anchored at xG sums:
λ₁ = max(home_xg − λ₃, ε), λ₂ = max(away_xg − λ₃, ε).

**Key properties:**
- Allows negative effective covariance even with λ₃ = 0 (diagonal
  inflation shifts mass from off-diagonal to diagonal cells).
- Models excess draws explicitly — soccer draws are more frequent than
  pure Poisson independence predicts.
- Reduces to Double Poisson when p → 0 (as Bundesliga shows).

### 1.3 Fitted parameters per league

| League | λ₃ | p | θ | ρ (Dixon-Coles) |
|---|---|---|---|---|
| Premier League (epl_1516) | 0.0000 | 0.0558 | 1.3645 | −0.0775 |
| Bundesliga (bundesliga_1516) | 0.0000 | ≈0.0001 | — | −0.0647 |
| La Liga (laliga_1516) | 0.0000 | 0.0145 | 1.4005 | −0.1003 |
| Serie A (seriea_1516) | 0.0000 | 0.0113 | 1.7542 | +0.0318 |
| Ligue 1 (ligue1_1516) | 0.0158 | 0.0121 | 1.4083 | −0.1100 |

**Interpretation:** All λ₃ = 0 (plain BP covariance still not supported by
data). EPL shows the strongest draw inflation (p = 5.6%), centered near θ
≈ 1.4 goals per team (close to the 1:1 result). Other leagues have small
but non-zero p (1.1–1.5%), except Bundesliga which degenerates to DP.
The θ values (1.4–1.75) indicate draw inflation concentrated at 1:1 and 2:2
scorelines, which are the modal draw results. Dixon-Coles still fits negative
ρ in four of five leagues, confirming systematic under-prediction of
low-scoring draws by Poisson independence.

---

## 2. Updated numerical results

All numbers pooled across the five leagues (N = 1822 matches).

### 2.1 Outcome probabilities (Brier / RPS)

| Approach | Brier | 95% CI | RPS | 95% CI |
|---|---|---|---|---|
| Poisson Binomial | **0.5391** | [0.520, 0.557] | **0.1768** | [0.170, 0.184] |
| **Bivariate Poisson** (DIBP) | **0.5405** | [0.524, 0.557] | **0.1774** | [0.171, 0.184] |
| Double Poisson | 0.5408 | [0.523, 0.558] | 0.1775 | [0.171, 0.184] |
| Dixon-Coles | 0.5407 | [0.523, 0.558] | 0.1775 | [0.171, 0.184] |
| Betting Odds | 0.5871 | [0.573, 0.601] | 0.1992 | [0.193, 0.205] |

**Change from old results:** "Bivariate Poisson" was 0.541/0.178 (identical
to DP). Now it is 0.5405/0.1774, slightly better than both DP and DC,
though still within overlapping confidence intervals.

### 2.2 Scoreline-distribution TVD

| Distribution | Poisson Binomial | Bivariate Poisson (DIBP) | Double Poisson | Dixon-Coles |
|---|---|---|---|---|
| Home goals | **0.036** | 0.083 | 0.085 | 0.085 |
| Away goals | **0.027** | 0.084 | 0.087 | 0.087 |
| Total goals | **0.054** | 0.098 | 0.099 | 0.094 |
| Exact scorelines | **0.064** | **0.124** | 0.129 | 0.123 |

### 2.3 Scoreline-distribution KL divergence

| Distribution | Poisson Binomial | Bivariate Poisson (DIBP) | Double Poisson | Dixon-Coles |
|---|---|---|---|---|
| Home goals | **0.007** | 0.021 | 0.022 | 0.022 |
| Away goals | **0.004** | 0.019 | 0.020 | 0.020 |
| Total goals | **0.012** | 0.031 | 0.030 | 0.030 |
| Exact scorelines | **0.023** | **0.053** | 0.055 | 0.054 |

**Important observation:** DIBP now differs meaningfully from DP:
- Exact scoreline TVD: 0.124 vs 0.129 (4% improvement)
- DIBP and Dixon-Coles are essentially tied on exact scorelines (DIBP
  slightly worse on TVD: 0.124 vs 0.123; slightly better on KL: 0.053 vs 0.054)
- DIBP home/away marginals differ slightly from DP (0.083/0.084 vs
  0.085/0.087) because diagonal inflation is not marginal-preserving;
  this is a small but real difference.

---

## 3. Required changes in the manuscript

### 3.1 Methods section — Bivariate Poisson paragraph (lines 156–186)

**Current text (lines 156–186):**
```
**Bivariate Poisson** [@karlis2003]: Home and away goals are represented as
H = W₁ + W₃, A = W₂ + W₃,
where W_i ∼ Pois(λ_i). The shared component W₃ induces covariance
Cov(H,A) = λ₃ ≥ 0. Marginal means are anchored via
λ₁ = max(λ_H−λ₃,ε), λ₂ = max(λ_A−λ₃,ε).
The shared component λ₃ is fitted by maximum likelihood separately for each league.

...

The Bivariate Poisson model is also anchored to the same Poisson marginals
by construction, but differs in the induced joint dependence structure.
Fitted league-level parameters for the Bivariate Poisson and Dixon–Coles
models are reported in the Supplementary Material.
```

**Required changes:**

(a) **Expand the BP description** to mention DIBP and the draw-inflation
    extension. The citation should include both @karlis2003 and @karlis2005.
    Replace the single-λ₃ description with the DIBP formulation:

    - Keep the base BP representation (W₁, W₂, W₃).
    - Add: "Because Cov(H,A) = λ₃ ≥ 0 cannot capture the near-zero or
      slightly negative home/away goal correlation found in most seasons,
      we employ the diagonal-inflated extension [@karlis2005], which adds a
      Poisson(θ) draw-inflation term:
        f_DIBP(x,y) = (1−p)·f_BP(x,y | λ₁,λ₂,λ₃)               for x≠y
        f_DIBP(x,y) = (1−p)·f_BP + p·Pois(x|θ)                   for x=y
      The three parameters (λ₃, p, θ) are fitted jointly per league by
      maximum likelihood."
    - Update the sentence about marginal preservation: "Unlike Double Poisson
      and Dixon--Coles, DIBP's diagonal inflation is not strictly marginal-
      preserving; the home and away goal marginals differ slightly from the
      pure Poisson marginals anchored at λ_H and λ_A."

(b) **Update the closing sentence** from "The shared component λ₃ is fitted
    by maximum likelihood separately for each league" to "The parameters
    (λ₃, p, θ) are fitted jointly by maximum likelihood separately for each
    league." The citation should now read [@karlis2003; @karlis2005].

(c) **Remove or update the comment at line 320:**
    `<!-- potentially adjust previous sentence if biv pois is adjusted -->`
    The sentence it refers to ("Further extensions include dependence
    structures that admit negative score correlation, such as copula-based
    models [@michels2024]") is now partly addressed — DIBP does allow
    effective negative covariance. Consider softening to: "Further extensions
    include richer dependence structures, such as copula-based models [@michels2024],
    and evaluation across additional seasons and xG providers."

### 3.2 Results — Outcome probabilities table (lines 231–238)

**Current table (in manuscript):**

| Approach          | Brier (95% CI)          | RPS (95% CI)             |
| Bivariate Poisson | 0.541 [0.524, 0.559]    | 0.178 [0.171, 0.184]     |

**Replace with:**

| Approach          | Brier (95% CI)          | RPS (95% CI)             |
| Bivariate Poisson | 0.541 [0.524, 0.557]    | 0.177 [0.171, 0.184]     |

*(Values rounded to 3 decimal places as in the table; Brier row value 0.541
is unchanged after rounding, but the upper CI bound rounds to 0.557 from 0.559.
The raw values are Brier=0.5405, RPS=0.1774.)*

The text at line 240 currently reads:
> "The three Poisson-based approaches achieve equal performance and the
> improvement through the Poisson Binomial is small, not extending beyond
> the 95% confidence intervals."

**Update to:**
> "The Poisson Binomial achieves modestly lower scores than the other three
> xG-based approaches; differences are small and confidence intervals remain
> largely overlapping. Bivariate Poisson performs slightly better than both
> Double Poisson and Dixon--Coles on Brier score, consistent with its explicit
> draw-inflation mechanism, though the differences are not statistically
> distinguishable."

### 3.3 Results — TVD table (lines 254–260)

**Current table:**

| Approach          | Home goals | Away goals | Total goals | Exact scorelines |
| Bivariate Poisson | 0.085      | 0.087      | 0.100       | 0.129            |

**Replace with:**

| Approach          | Home goals | Away goals | Total goals | Exact scorelines |
| Bivariate Poisson | 0.083      | 0.084      | 0.098       | 0.124            |

### 3.4 Results — KL table (lines 262–268)

**Current table:**

| Approach          | Home goals | Away goals | Total goals | Exact scorelines |
| Bivariate Poisson | 0.022      | 0.020      | 0.031       | 0.055            |

**Replace with:**

| Approach          | Home goals | Away goals | Total goals | Exact scorelines |
| Bivariate Poisson | 0.021      | 0.019      | 0.031       | 0.053            |

*(Total goals KL is unchanged at 0.031. Home/away marginals change because
DIBP is not exactly marginal-preserving.)*

### 3.5 Results — Narrative text (lines 273–276)

**Current text (lines 273–275):**
> "On exact scorelines, it reduces TVD from 0.123 for the best Poisson-family
> comparator to 0.064, and KL divergence from 0.054 to 0.023."

**Update to:**
> "On exact scorelines, it reduces TVD from 0.124 for the best Poisson-family
> comparator to 0.064, and KL divergence from 0.053 to 0.023."

*(The "best Poisson-family comparator" changes from Dixon-Coles TVD 0.123 to
DIBP TVD 0.124, since now DIBP slightly edges out DC on TVD.)*

Wait — check the values: DIBP TVD=0.124, DC TVD=0.123. So DC is still the
best Poisson-family TVD. Retain "0.123 for the best Poisson-family
comparator" only if referring to DC. Better phrasing:

> "On exact scorelines, it reduces TVD from 0.123 (Dixon-Coles, the best
> Poisson-family comparator) to 0.064, and KL divergence from 0.053
> (Bivariate Poisson, the best Poisson-family comparator on KL) to 0.023."

**Current text (lines 275–276):**
> "The three Poisson-family models have identical home- and away-goal marginal
> fits because all are anchored to the same Poisson marginal distributions.
> They differ only in their joint scoreline structure. Dixon--Coles provides
> the strongest improvement among the Poisson-family models on exact scorelines,
> consistent with its low-score correction. Bivariate Poisson provides
> negligible improvement over Double Poisson..."

**Update to:**
> "Double Poisson and Dixon--Coles have identical home- and away-goal marginal
> fits because both are anchored to the same Poisson marginal distributions.
> The Bivariate Poisson model differs slightly because its diagonal inflation
> is not marginal-preserving, yielding marginally lower home- and away-goal TVD.
> On exact scorelines, Bivariate Poisson (TVD 0.124) and Dixon-Coles (TVD 0.123)
> are essentially tied, both providing a meaningful improvement over Double Poisson
> (TVD 0.129), consistent with their respective mechanisms for correcting
> draw probabilities."

### 3.6 Discussion — Key paragraph on BP (lines 295–297)

**Current text (lines 295–297):**
> "The Bivariate Poisson is essentially indistinguishable from Double Poisson:
> the fitted shared-component parameter λ₃ is effectively zero in three of
> five leagues (Bundesliga, La Liga, and Serie A: λ₃ = 0.0002), with only
> EPL (λ₃ = 0.0144) and Ligue 1 (λ₃ = 0.0205) departing marginally from zero.
> The near-zero or slightly negative empirical home/away goal correlation in
> this sample consistently pushes maximum likelihood toward λ₃ → 0, confirmed
> by negative ρ estimates from Dixon--Coles in four of five leagues. This
> reflects not a flaw in the Bivariate Poisson framework but rather that the
> positive-covariance structure it encodes is not supported by this dataset."

**This paragraph needs substantial revision.** The DIBP does differ from DP.
Replace with something like:

> "The Bivariate Poisson model, extended with Poisson diagonal draw inflation
> [@karlis2005], provides a modest improvement over Double Poisson on exact
> scorelines (TVD 0.124 vs 0.129). This improvement is driven by the draw-
> inflation weight p, which is meaningfully above zero in four of five leagues
> (EPL: p = 5.6%; La Liga: p = 1.5%; Serie A: p = 1.1%; Ligue 1: p = 1.2%),
> while the Bundesliga shows essentially no draw excess (p ≈ 0). The covariance
> parameter λ₃ is effectively zero in all five leagues, confirming that the
> near-zero or slightly negative empirical home/away goal correlation does not
> support the plain Bivariate Poisson covariance structure. The draw-inflation
> mechanism compensates for this by explicitly modelling the documented excess
> draw frequency in soccer, effectively allowing the model to place additional
> probability mass on x:x scorelines. The resulting exact-scoreline fit is
> essentially equivalent to Dixon--Coles (TVD 0.124 vs 0.123), suggesting that
> draw correction — whether through diagonal inflation or through Dixon–Coles's
> low-score multiplicative adjustment — is the more effective lever for
> improving scoreline-distribution fit than the covariance structure alone."

### 3.7 Supplementary material note on Bivariate Poisson parameters (line 186)

**Current text (line 186):**
> "Fitted league-level parameters for the Bivariate Poisson and Dixon–Coles
> models are reported in the Supplementary Material."

**Update to:**
> "Fitted league-level parameters for the Bivariate Poisson model (λ₃, p, θ)
> and the Dixon–Coles correction (ρ) are reported in the Supplementary Material."

### 3.8 Related work section (line 65–66) — already correct

The related work section at line 65 already reads:
> "The diagonal-inflated Bivariate Poisson [@karlis2005] allowed for negative
> correlation and overall improved draw calibration compared to empirical
> observations."

This is accurate and does not need changing. The @karlis2005 citation is
already present.

### 3.9 Conclusion (lines 326–327)

**Current text:**
> "the Poisson Binomial substantially reduced Total Variation Distance and
> Kullback--Leibler divergence relative to Double Poisson, Dixon--Coles, and
> Bivariate Poisson models."

This remains accurate — no change needed.

---

## 4. New/updated citations required

Ensure `references.bib` (at `manuscript/references.bib`) contains:

```bibtex
@article{karlis2005,
  ...
}
```

The key @karlis2005 is already cited in Related Work (line 65). If the
`references.bib` entry is missing, locate the full record in
`/home/max/drive/phd_library.bib` (which is the master bibliography).

---

## 5. No changes required

The following sections are correct as-is and do not need updating:
- Abstract — describes "Bivariate Poisson" without specific numerical values
  from BP results; remains accurate.
- Introduction (line 57) — describes comparing "Bivariate Poisson" as one
  of the models; remains accurate.
- Related work (lines 65–66) — already cites @karlis2005 correctly.
- Sample statistics (lines 223–225) — not model-dependent.
- Bet365/calibration comparison (lines 298–303) — not affected by BP→DIBP.
- Applications section (lines 305–309) — not affected.
- Limitations (lines 313–319) — limitation about independent Bernoulli
  assumption still holds.
- Conclusion first paragraph (lines 324–325) — not affected.

---

## 6. How to verify after making changes

1. Check that all three "Bivariate Poisson" rows in `@tbl-evaluation-metrics`,
   `@tbl-tvd-results`, and `@tbl-kl-results` use the DIBP numbers above.
2. The figures embedded from `02_evaluation.ipynb` are already regenerated
   with the label "Bivariate Poisson" (the pipeline uses DIBP internally but
   displays as "Bivariate Poisson"); no figure changes needed.
3. Re-render the manuscript with `quarto render polynomial-multiplication-xg.qmd`
   and confirm no broken references.

---

## 7. Quick reference: old vs new BP values

| Metric | Old (plain BP) | New (DIBP) | Change |
|---|---|---|---|
| Brier | 0.541 | 0.541 (rounds same) | Negligible at 3 dp |
| RPS | 0.178 | 0.177 | −0.001 |
| TVD home goals | 0.085 | 0.083 | −0.002 |
| TVD away goals | 0.087 | 0.084 | −0.003 |
| TVD total goals | 0.100 | 0.098 | −0.002 |
| TVD exact scorelines | 0.129 | 0.124 | −0.005 (-4%) |
| KL home goals | 0.022 | 0.021 | −0.001 |
| KL away goals | 0.020 | 0.019 | −0.001 |
| KL total goals | 0.031 | 0.031 | 0 |
| KL exact scorelines | 0.055 | 0.053 | −0.002 |
| Discussion claim | "BP ≈ DP" | "BP ≈ DC on scorelines" | Substantial change |
