"""
probability_generation.py
=========================
Module B: Generate match-outcome and exact-score probability distributions.

Public API
----------
Core algorithms (competition- and match-id-agnostic)
    poisson_binomial_pmf(xg_values)
    double_poisson_pmf(lambda_val, max_goals)
    bivariate_poisson_matrix(lambda1, lambda2, lambda3, max_goals)
    diagonal_inflated_bivariate_poisson_matrix(lambda1, lambda2, lambda3, p, theta, max_goals)
    dixon_coles_matrix(lambda_h, lambda_a, rho, max_goals)
    scoreline_matrix_from_pmfs(home_pmf, away_pmf)
    outcome_probs_from_scoreline(scoreline_matrix)

Parameter fitting (global per dataset, via MLE on observed scores)
    fit_bivariate_poisson_lambda3(data, max_goals)
    fit_dibp_params(data, max_goals)
    fit_dixon_coles_rho(data, max_goals)

Per-approach generators  (accept a match-summary DataFrame + optional filters)
    generate_poisson_binomial_probs(data, ...)   → outcomes DataFrame
    generate_double_poisson_probs(data, ...)     → outcomes DataFrame
    generate_dibp_probs(data, ...)               → outcomes DataFrame
    generate_dixon_coles_probs(data, ...)        → outcomes DataFrame
    generate_betting_odds_probs(data, ...)       → outcomes DataFrame
    generate_poisson_binomial_score_dists(data, ...)  → scores DataFrame
    generate_double_poisson_score_dists(data, ...)    → scores DataFrame
    generate_dibp_score_dists(data, ...)              → scores DataFrame
    generate_dixon_coles_score_dists(data, ...)       → scores DataFrame

Persistence
    save_probabilities(df, output_dir, competition_slug, approach, target)
    load_probabilities(path, parse_distributions=False)

Output schemas
--------------
Outcomes parquet  (…_outcomes.parquet):
    match_id, match_date, competition_name, season_name,
    home_team, away_team, home_goals, away_goals,
    actual_outcome, p_home_win, p_draw, p_away_win,
    approach

Scores parquet  (…_scores.parquet):
    match_id, match_date, competition_name, season_name,
    home_team, away_team, home_goals, away_goals,
    home_xg, away_xg,
    home_pmf (JSON), away_pmf (JSON), scoreline_matrix (JSON),
    approach
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import minimize_scalar, minimize


# ── Core algorithms ───────────────────────────────────────────────────────────

def poisson_binomial_pmf(xg_values: Sequence[float]) -> np.ndarray:
    """
    Poisson Binomial PMF via polynomial multiplication (Ruiz et al., 2015).

    Each shot is an independent Bernoulli trial with success probability p_i.
    The resulting polynomial P(a) = ∏ [(1-p_i) + p_i·a] has coefficients
    equal to P(X = k) for k = 0, 1, …, n.

    Parameters
    ----------
    xg_values : sequence of float
        Per-shot xG values (probabilities in (0, 1]).

    Returns
    -------
    np.ndarray
        PMF array of length n+1 where element k = P(X = k).
    """
    xg = [float(x) for x in xg_values if not np.isnan(x)]
    if not xg:
        return np.array([1.0])
    result = np.array([1.0 - xg[0], xg[0]])
    for p in xg[1:]:
        result = np.convolve(result, np.array([1.0 - p, p]))
    return result


def double_poisson_pmf(lambda_val: float, max_goals: int = 10) -> np.ndarray:
    """
    Truncated and renormalised Poisson PMF.

    Parameters
    ----------
    lambda_val : float
        Poisson rate (total team xG).
    max_goals : int
        Maximum goals to include; PMF is truncated and renormalised.

    Returns
    -------
    np.ndarray of length max_goals + 1.
    """
    pmf = scipy_stats.poisson.pmf(np.arange(max_goals + 1), lambda_val)
    return pmf / pmf.sum()


def bivariate_poisson_matrix(
    lambda1: float,
    lambda2: float,
    lambda3: float,
    max_goals: int = 10,
) -> np.ndarray:
    """
    Joint PMF matrix of the Bivariate Poisson (Karlis & Ntzoufras, 2003).

    Construction: X = W₁ + W₃, Y = W₂ + W₃, with W_i ~ Pois(λ_i) independent.
    Marginals: X ~ Pois(λ₁ + λ₃), Y ~ Pois(λ₂ + λ₃).  Cov(X, Y) = λ₃ ≥ 0.

    Joint PMF:
        P(X=x, Y=y) = e^{-(λ₁+λ₂+λ₃)}
                      · Σ_{k=0}^{min(x,y)} (λ₃^k/k!) (λ₁^{x-k}/(x-k)!) (λ₂^{y-k}/(y-k)!)

    Parameters
    ----------
    lambda1, lambda2, lambda3 : float
        Components of the Bivariate Poisson.  Each must be ≥ 0.
    max_goals : int
        Goal cap.  The (max_goals+1) × (max_goals+1) matrix is renormalised
        to sum to 1 to absorb truncation mass.

    Returns
    -------
    np.ndarray of shape (max_goals+1, max_goals+1) with [h, a] = P(home=h, away=a).
    """
    n = max_goals + 1
    l1 = max(float(lambda1), 0.0)
    l2 = max(float(lambda2), 0.0)
    l3 = max(float(lambda3), 0.0)

    p1 = scipy_stats.poisson.pmf(np.arange(n), max(l1, 1e-12))
    p2 = scipy_stats.poisson.pmf(np.arange(n), max(l2, 1e-12))

    # Independence shortcut
    if l3 == 0.0:
        mat = np.outer(p1, p2)
        s = mat.sum()
        return mat / s if s > 0 else mat

    p3 = scipy_stats.poisson.pmf(np.arange(n), l3)

    matrix = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            kmax = min(x, y)
            # Σ_{k=0}^{kmax} P(W₃=k) · P(W₁=x-k) · P(W₂=y-k)
            ks = np.arange(kmax + 1)
            matrix[x, y] = float(np.sum(p3[ks] * p1[x - ks] * p2[y - ks]))

    s = matrix.sum()
    return matrix / s if s > 0 else matrix


def diagonal_inflated_bivariate_poisson_matrix(
    lambda1: float,
    lambda2: float,
    lambda3: float,
    p: float,
    theta: float,
    max_goals: int = 10,
) -> np.ndarray:
    """
    Joint PMF of the Diagonal Inflated Bivariate Poisson (Karlis & Ntzoufras, 2005).

    Mixes a base Bivariate Poisson with a Poisson(θ) diagonal inflation component:

        f_IBP(x,y) = (1−p)·f_BP(x,y | λ₁,λ₂,λ₃)                   x ≠ y
        f_IBP(x,y) = (1−p)·f_BP(x,y | λ₁,λ₂,λ₃) + p·Pois(x | θ)   x = y

    This allows negative covariance (unlike plain BP which constrains λ₃ ≥ 0)
    and explicitly models the excess draw frequency observed in soccer.

    Parameters
    ----------
    lambda1, lambda2, lambda3 : float
        Bivariate Poisson components.  λ₁, λ₂ are the independent rates;
        λ₃ is the shared (covariance) component.
    p : float
        Mixing proportion for diagonal inflation; in [0, 1].
    theta : float
        Rate of the Poisson diagonal-inflation distribution; > 0.
    max_goals : int
        Goal cap; matrix is renormalised after truncation.

    Returns
    -------
    np.ndarray of shape (max_goals+1, max_goals+1).
    """
    n = max_goals + 1
    p     = float(np.clip(p, 0.0, 1.0))
    theta = max(float(theta), 1e-12)

    bp        = bivariate_poisson_matrix(lambda1, lambda2, lambda3, max_goals)
    inflation = scipy_stats.poisson.pmf(np.arange(n), theta)
    s_inf     = inflation.sum()
    if s_inf > 0:
        inflation /= s_inf  # renormalise for truncation at max_goals

    matrix = (1.0 - p) * bp
    for k in range(n):
        matrix[k, k] += p * inflation[k]

    s = matrix.sum()
    return matrix / s if s > 0 else matrix


def dixon_coles_matrix(
    lambda_h: float,
    lambda_a: float,
    rho: float,
    max_goals: int = 10,
) -> np.ndarray:
    """
    Joint PMF matrix of the Dixon–Coles model (Dixon & Coles, 1997).

    Adjusts the independent-Poisson product on the four low-score cells:

        τ(0,0) = 1 − λ_h λ_a ρ
        τ(0,1) = 1 + λ_h ρ
        τ(1,0) = 1 + λ_a ρ
        τ(1,1) = 1 − ρ
        τ(x,y) = 1 otherwise

    Negative τ values (indicating a ρ outside the feasible region for the
    given λ pair) are clipped to 0 before renormalisation.

    Parameters
    ----------
    lambda_h, lambda_a : float
        Home/away scoring rates (e.g. team xG totals).
    rho : float
        Dependence parameter.  Negative ρ inflates 0:0 / 1:1 and shrinks
        1:0 / 0:1 — the empirically observed adjustment in football.
    max_goals : int
        Goal cap; matrix is renormalised after τ adjustment + truncation.

    Returns
    -------
    np.ndarray of shape (max_goals+1, max_goals+1).
    """
    n = max_goals + 1
    lh = max(float(lambda_h), 1e-12)
    la = max(float(lambda_a), 1e-12)

    p_h = scipy_stats.poisson.pmf(np.arange(n), lh)
    p_a = scipy_stats.poisson.pmf(np.arange(n), la)
    matrix = np.outer(p_h, p_a)

    matrix[0, 0] *= (1.0 - lh * la * rho)
    matrix[0, 1] *= (1.0 + lh * rho)
    matrix[1, 0] *= (1.0 + la * rho)
    matrix[1, 1] *= (1.0 - rho)

    matrix = np.clip(matrix, 0.0, None)
    s = matrix.sum()
    return matrix / s if s > 0 else matrix


def scoreline_matrix_from_pmfs(
    home_pmf: np.ndarray,
    away_pmf: np.ndarray,
) -> np.ndarray:
    """
    Joint scoreline probability matrix under the independence assumption.

    Returns an (n+1) × (n+1) matrix where entry [h, a] = P(home=h) × P(away=a).
    """
    return np.outer(home_pmf, away_pmf)


def outcome_probs_from_scoreline(
    scoreline_matrix: np.ndarray,
) -> dict[str, float]:
    """
    Aggregate a scoreline matrix into (home_win, draw, away_win) probabilities.
    """
    p_home_win = float(np.sum(np.tril(scoreline_matrix, k=-1)))
    p_draw     = float(np.sum(np.diag(scoreline_matrix)))
    p_away_win = float(np.sum(np.triu(scoreline_matrix, k=1)))
    return {"p_home_win": p_home_win, "p_draw": p_draw, "p_away_win": p_away_win}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pad_norm(pmf: np.ndarray, length: int) -> np.ndarray:
    """Pad to `length`, truncate if too long, then renormalise."""
    pmf = np.pad(pmf, (0, max(0, length - len(pmf))))[:length]
    s = pmf.sum()
    return pmf / s if s > 0 else pmf


def _filter_data(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]],
    match_ids: Optional[Sequence[int]],
) -> pd.DataFrame:
    df = data.copy()
    if competition_names is not None:
        df = df[df["competition_name"].isin(competition_names)]
    if match_ids is not None:
        df = df[df["match_id"].isin(match_ids)]
    return df


def _actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    elif home_goals < away_goals:
        return "away_win"
    return "draw"


def _marginals_from_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (home_pmf, away_pmf) from a joint scoreline matrix."""
    return matrix.sum(axis=1), matrix.sum(axis=0)


# ── Parameter fitting (global per dataset, MLE on observed scores) ───────────

def fit_bivariate_poisson_lambda3(
    data: pd.DataFrame,
    max_goals: int = 10,
    bracket: tuple[float, float] = (1e-4, 1.0),
) -> float:
    """
    Fit the Bivariate Poisson covariance parameter λ₃ globally via MLE.

    For each match the marginals are anchored at the observed xG sums:
        λ₁ = max(home_xg − λ₃, ε),   λ₂ = max(away_xg − λ₃, ε)
    so that for matches with one team's xG below λ₃ the model gracefully
    degrades to (near-)independence rather than producing negative rates.

    Returns
    -------
    float — the λ₃ ≥ 0 that maximises the joint log-likelihood of observed
    (home_goals, away_goals) on ``data``.
    """
    home_xg = data["home_xg"].astype(float).to_numpy()
    away_xg = data["away_xg"].astype(float).to_numpy()
    home_g  = data["home_goals"].clip(upper=max_goals).astype(int).to_numpy()
    away_g  = data["away_goals"].clip(upper=max_goals).astype(int).to_numpy()

    def neg_log_lik(lambda3: float) -> float:
        lambda3 = max(float(lambda3), 0.0)
        ll = 0.0
        for hxg, axg, hg, ag in zip(home_xg, away_xg, home_g, away_g):
            l1 = max(hxg - lambda3, 1e-6)
            l2 = max(axg - lambda3, 1e-6)
            mat = bivariate_poisson_matrix(l1, l2, lambda3, max_goals)
            ll += np.log(max(mat[hg, ag], 1e-12))
        return -ll

    lo, hi = bracket
    # Cap upper bound below the smallest xG so most matches can use the rate
    # without falling back to clipping.
    upper = max(min(home_xg.min(), away_xg.min()) - 1e-3, lo + 1e-3)
    upper = min(upper, hi)
    if upper <= lo:
        return 0.0
    res = minimize_scalar(neg_log_lik, bounds=(lo, upper), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def fit_dibp_params(
    data: pd.DataFrame,
    max_goals: int = 10,
    lambda3_bound: tuple[float, float] = (0.0, 1.0),
    p_bound: tuple[float, float] = (1e-4, 0.5),
    theta_bound: tuple[float, float] = (1e-4, 3.0),
) -> tuple[float, float, float]:
    """
    Fit DIBP parameters (λ₃, p, θ) jointly via MLE (L-BFGS-B).

    Per-match marginals are anchored at observed xG sums:
        λ₁ = max(home_xg − λ₃, ε),   λ₂ = max(away_xg − λ₃, ε)

    Nine starting points (3 values of p₀ × 3 values of θ₀) guard against
    local optima; the one with the highest log-likelihood is returned.

    Returns
    -------
    (lambda3, p, theta) — all floats ≥ 0.
    """
    home_xg = data["home_xg"].astype(float).to_numpy()
    away_xg = data["away_xg"].astype(float).to_numpy()
    home_g  = data["home_goals"].clip(upper=max_goals).astype(int).to_numpy()
    away_g  = data["away_goals"].clip(upper=max_goals).astype(int).to_numpy()

    l3_hi = max(min(home_xg.min(), away_xg.min()) - 1e-3, 1e-4)
    l3_hi = min(l3_hi, lambda3_bound[1])
    bounds = [(lambda3_bound[0], l3_hi), p_bound, theta_bound]

    def neg_log_lik(params: np.ndarray) -> float:
        l3, p, theta = params
        l3    = max(float(l3),    0.0)
        p     = float(np.clip(p,  1e-9, 1 - 1e-9))
        theta = max(float(theta), 1e-9)
        ll = 0.0
        for hxg, axg, hg, ag in zip(home_xg, away_xg, home_g, away_g):
            l1  = max(hxg - l3, 1e-6)
            l2  = max(axg - l3, 1e-6)
            mat = diagonal_inflated_bivariate_poisson_matrix(l1, l2, l3, p, theta, max_goals)
            ll += np.log(max(mat[hg, ag], 1e-12))
        return -ll

    best: object = None
    for p0 in [0.05, 0.10, 0.20]:
        for th0 in [0.5, 1.0, 1.5]:
            res = minimize(
                neg_log_lik, x0=[1e-3, p0, th0], bounds=bounds,
                method="L-BFGS-B", options={"ftol": 1e-8, "gtol": 1e-6},
            )
            if best is None or res.fun < best.fun:
                best = res

    l3, p, theta = best.x
    return float(max(l3, 0.0)), float(np.clip(p, 0.0, 1.0)), float(max(theta, 0.0))


def fit_dixon_coles_rho(
    data: pd.DataFrame,
    max_goals: int = 10,
    bracket: tuple[float, float] = (-0.3, 0.3),
) -> float:
    """
    Fit the Dixon–Coles dependence parameter ρ globally via MLE.

    Returns
    -------
    float — the ρ that maximises the joint log-likelihood of observed
    (home_goals, away_goals) under the Dixon–Coles model with per-match
    rates λ_h = home_xg, λ_a = away_xg.
    """
    home_xg = data["home_xg"].astype(float).to_numpy()
    away_xg = data["away_xg"].astype(float).to_numpy()
    home_g  = data["home_goals"].clip(upper=max_goals).astype(int).to_numpy()
    away_g  = data["away_goals"].clip(upper=max_goals).astype(int).to_numpy()

    def neg_log_lik(rho: float) -> float:
        ll = 0.0
        for hxg, axg, hg, ag in zip(home_xg, away_xg, home_g, away_g):
            mat = dixon_coles_matrix(hxg, axg, rho, max_goals)
            ll += np.log(max(mat[hg, ag], 1e-12))
        return -ll

    res = minimize_scalar(neg_log_lik, bounds=bracket, method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


_META_OUTCOME_COLS = [
    "match_id", "match_date", "competition_name", "season_name",
    "home_team", "away_team", "home_goals", "away_goals",
]

_META_SCORE_COLS = [
    "match_id", "match_date", "competition_name", "season_name",
    "home_team", "away_team", "home_goals", "away_goals",
    "home_xg", "away_xg",
]


# ── Outcome generators ────────────────────────────────────────────────────────

def generate_poisson_binomial_probs(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]] = None,
    match_ids: Optional[Sequence[int]] = None,
    max_goals: int = 10,
) -> pd.DataFrame:
    """
    Generate outcome probabilities using Poisson Binomial (polynomial multiplication).

    Requires ``home_xg_values`` and ``away_xg_values`` list columns.

    Parameters
    ----------
    data : pd.DataFrame
        Match-summary DataFrame as returned by ``load_competition_data()``.
    competition_names : sequence of str, optional
        Restrict to these competition names.
    match_ids : sequence of int, optional
        Restrict to these match IDs.
    max_goals : int
        Goals cap for probability truncation.

    Returns
    -------
    pd.DataFrame with outcome schema + ``approach = 'poisson_binomial'``.
    """
    df = _filter_data(data, competition_names, match_ids)
    rows: list[dict] = []
    for _, row in df.iterrows():
        home_xg = [x for x in row["home_xg_values"] if pd.notna(x)]
        away_xg = [x for x in row["away_xg_values"] if pd.notna(x)]
        home_pmf = _pad_norm(poisson_binomial_pmf(home_xg), max_goals + 1)
        away_pmf = _pad_norm(poisson_binomial_pmf(away_xg), max_goals + 1)
        mat   = scoreline_matrix_from_pmfs(home_pmf, away_pmf)
        probs = outcome_probs_from_scoreline(mat)
        rec   = {c: row.get(c) for c in _META_OUTCOME_COLS}
        rec.update(probs)
        rec["actual_outcome"] = _actual_outcome(int(row["home_goals"]), int(row["away_goals"]))
        rec["approach"] = "poisson_binomial"
        rows.append(rec)
    return pd.DataFrame(rows)


def generate_double_poisson_probs(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]] = None,
    match_ids: Optional[Sequence[int]] = None,
    max_goals: int = 10,
) -> pd.DataFrame:
    """
    Generate outcome probabilities using the Double Poisson model.

    Uses total team xG as the Poisson rate (λ_home = home_xg, λ_away = away_xg).

    Returns pd.DataFrame with outcome schema + ``approach = 'double_poisson'``.
    """
    df = _filter_data(data, competition_names, match_ids)
    rows: list[dict] = []
    for _, row in df.iterrows():
        home_pmf = double_poisson_pmf(float(row["home_xg"]), max_goals)
        away_pmf = double_poisson_pmf(float(row["away_xg"]), max_goals)
        mat   = scoreline_matrix_from_pmfs(home_pmf, away_pmf)
        probs = outcome_probs_from_scoreline(mat)
        rec   = {c: row.get(c) for c in _META_OUTCOME_COLS}
        rec.update(probs)
        rec["actual_outcome"] = _actual_outcome(int(row["home_goals"]), int(row["away_goals"]))
        rec["approach"] = "double_poisson"
        rows.append(rec)
    return pd.DataFrame(rows)


def generate_bivariate_poisson_probs(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]] = None,
    match_ids: Optional[Sequence[int]] = None,
    max_goals: int = 10,
    lambda3: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate outcome probabilities using the Bivariate Poisson model
    (Karlis & Ntzoufras, 2003) with per-match rates anchored at xG sums.

    The shared component λ₃ (= Cov(home, away)) is fitted globally on
    ``data`` by MLE on observed scores unless supplied via ``lambda3``.
    For each match: λ₁ = max(home_xg − λ₃, ε), λ₂ = max(away_xg − λ₃, ε).

    Returns pd.DataFrame with outcome schema + ``approach = 'bivariate_poisson'``.
    """
    df = _filter_data(data, competition_names, match_ids)
    if lambda3 is None:
        lambda3 = fit_bivariate_poisson_lambda3(df, max_goals)
    rows: list[dict] = []
    for _, row in df.iterrows():
        l1 = max(float(row["home_xg"]) - lambda3, 1e-6)
        l2 = max(float(row["away_xg"]) - lambda3, 1e-6)
        mat = bivariate_poisson_matrix(l1, l2, lambda3, max_goals)
        probs = outcome_probs_from_scoreline(mat)
        rec = {c: row.get(c) for c in _META_OUTCOME_COLS}
        rec.update(probs)
        rec["actual_outcome"] = _actual_outcome(int(row["home_goals"]), int(row["away_goals"]))
        rec["approach"] = "bivariate_poisson"
        rows.append(rec)
    return pd.DataFrame(rows)


def generate_dibp_probs(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]] = None,
    match_ids: Optional[Sequence[int]] = None,
    max_goals: int = 10,
    lambda3: Optional[float] = None,
    p: Optional[float] = None,
    theta: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate outcome probabilities using the Diagonal Inflated Bivariate Poisson
    (Karlis & Ntzoufras, 2005).

    (λ₃, p, θ) are fitted jointly via MLE on ``data`` unless all three are
    supplied.  Per-match marginals are anchored at xG:
        λ₁ = max(home_xg − λ₃, ε),  λ₂ = max(away_xg − λ₃, ε)

    Returns pd.DataFrame with outcome schema + ``approach = 'diagonal_inflated_bivariate_poisson'``.
    """
    df = _filter_data(data, competition_names, match_ids)
    if lambda3 is None or p is None or theta is None:
        lambda3, p, theta = fit_dibp_params(df, max_goals)
    rows: list[dict] = []
    for _, row in df.iterrows():
        l1  = max(float(row["home_xg"]) - lambda3, 1e-6)
        l2  = max(float(row["away_xg"]) - lambda3, 1e-6)
        mat = diagonal_inflated_bivariate_poisson_matrix(l1, l2, lambda3, p, theta, max_goals)
        probs = outcome_probs_from_scoreline(mat)
        rec   = {c: row.get(c) for c in _META_OUTCOME_COLS}
        rec.update(probs)
        rec["actual_outcome"] = _actual_outcome(int(row["home_goals"]), int(row["away_goals"]))
        rec["approach"] = "diagonal_inflated_bivariate_poisson"
        rows.append(rec)
    return pd.DataFrame(rows)


def generate_dixon_coles_probs(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]] = None,
    match_ids: Optional[Sequence[int]] = None,
    max_goals: int = 10,
    rho: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate outcome probabilities using the Dixon–Coles double-Poisson
    adjustment (Dixon & Coles, 1997) with per-match rates λ_h = home_xg,
    λ_a = away_xg.

    The dependence parameter ρ is fitted globally on ``data`` by MLE on
    observed scores unless supplied via ``rho``.

    Returns pd.DataFrame with outcome schema + ``approach = 'dixon_coles'``.
    """
    df = _filter_data(data, competition_names, match_ids)
    if rho is None:
        rho = fit_dixon_coles_rho(df, max_goals)
    rows: list[dict] = []
    for _, row in df.iterrows():
        mat = dixon_coles_matrix(float(row["home_xg"]), float(row["away_xg"]),
                                 rho, max_goals)
        probs = outcome_probs_from_scoreline(mat)
        rec = {c: row.get(c) for c in _META_OUTCOME_COLS}
        rec.update(probs)
        rec["actual_outcome"] = _actual_outcome(int(row["home_goals"]), int(row["away_goals"]))
        rec["approach"] = "dixon_coles"
        rows.append(rec)
    return pd.DataFrame(rows)


def generate_betting_odds_probs(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]] = None,
    match_ids: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Extract pre-match betting odds as outcome probabilities.

    Requires ``betting_p_home``, ``betting_p_draw``, ``betting_p_away`` columns.
    Rows without odds are silently dropped.

    Returns pd.DataFrame with outcome schema + ``approach = 'betting_odds'``.
    """
    df = _filter_data(data, competition_names, match_ids)
    required = {"betting_p_home", "betting_p_draw", "betting_p_away"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data missing columns: {missing}")

    df = df.dropna(subset=list(required)).copy()
    rows: list[dict] = []
    for _, row in df.iterrows():
        rec = {c: row.get(c) for c in _META_OUTCOME_COLS}
        rec["p_home_win"]    = float(row["betting_p_home"])
        rec["p_draw"]        = float(row["betting_p_draw"])
        rec["p_away_win"]    = float(row["betting_p_away"])
        rec["actual_outcome"] = _actual_outcome(int(row["home_goals"]), int(row["away_goals"]))
        rec["approach"] = "betting_odds"
        rows.append(rec)
    return pd.DataFrame(rows)


# ── Score-distribution generators ────────────────────────────────────────────

def generate_poisson_binomial_score_dists(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]] = None,
    match_ids: Optional[Sequence[int]] = None,
    max_goals: int = 10,
) -> pd.DataFrame:
    """
    Generate per-match goal distributions using Poisson Binomial.

    Returns a DataFrame with columns:
        [meta cols] + home_pmf (JSON), away_pmf (JSON),
        scoreline_matrix (JSON), approach = 'poisson_binomial'
    """
    df = _filter_data(data, competition_names, match_ids)
    return _build_score_dists(df, "poisson_binomial", max_goals)


def generate_double_poisson_score_dists(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]] = None,
    match_ids: Optional[Sequence[int]] = None,
    max_goals: int = 10,
) -> pd.DataFrame:
    """
    Generate per-match goal distributions using Double Poisson.

    Returns a DataFrame with columns:
        [meta cols] + home_pmf (JSON), away_pmf (JSON),
        scoreline_matrix (JSON), approach = 'double_poisson'
    """
    df = _filter_data(data, competition_names, match_ids)
    return _build_score_dists(df, "double_poisson", max_goals)


def generate_bivariate_poisson_score_dists(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]] = None,
    match_ids: Optional[Sequence[int]] = None,
    max_goals: int = 10,
    lambda3: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate per-match goal distributions using the Bivariate Poisson model.

    Marginal home/away PMFs are extracted from the joint scoreline matrix
    (after τ-style truncation/renormalisation), so they correctly reflect
    the dependence induced by λ₃.

    Returns scores DataFrame with ``approach = 'bivariate_poisson'``.
    """
    df = _filter_data(data, competition_names, match_ids)
    if lambda3 is None:
        lambda3 = fit_bivariate_poisson_lambda3(df, max_goals)
    return _build_score_dists(df, "bivariate_poisson", max_goals, lambda3=lambda3)


def generate_dibp_score_dists(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]] = None,
    match_ids: Optional[Sequence[int]] = None,
    max_goals: int = 10,
    lambda3: Optional[float] = None,
    p: Optional[float] = None,
    theta: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate per-match goal distributions using the Diagonal Inflated Bivariate
    Poisson model.

    Marginal home/away PMFs are extracted from the joint scoreline matrix after
    the diagonal inflation, correctly reflecting both the covariance and draw
    excess modelled by (λ₃, p, θ).

    Returns scores DataFrame with ``approach = 'diagonal_inflated_bivariate_poisson'``.
    """
    df = _filter_data(data, competition_names, match_ids)
    if lambda3 is None or p is None or theta is None:
        lambda3, p, theta = fit_dibp_params(df, max_goals)
    return _build_score_dists(
        df, "diagonal_inflated_bivariate_poisson", max_goals,
        lambda3=lambda3, p=p, theta=theta,
    )


def generate_dixon_coles_score_dists(
    data: pd.DataFrame,
    competition_names: Optional[Sequence[str]] = None,
    match_ids: Optional[Sequence[int]] = None,
    max_goals: int = 10,
    rho: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate per-match goal distributions using the Dixon–Coles model.

    Marginal home/away PMFs are extracted from the joint matrix after
    the τ adjustment + renormalisation, so they reflect the small
    redistribution among the four low-score cells.

    Returns scores DataFrame with ``approach = 'dixon_coles'``.
    """
    df = _filter_data(data, competition_names, match_ids)
    if rho is None:
        rho = fit_dixon_coles_rho(df, max_goals)
    return _build_score_dists(df, "dixon_coles", max_goals, rho=rho)


def _build_score_dists(
    df: pd.DataFrame,
    approach: str,
    max_goals: int,
    lambda3: Optional[float] = None,
    p: Optional[float] = None,
    theta: Optional[float] = None,
    rho: Optional[float] = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in df.iterrows():
        if approach == "poisson_binomial":
            home_xg_vals = [x for x in row["home_xg_values"] if pd.notna(x)]
            away_xg_vals = [x for x in row["away_xg_values"] if pd.notna(x)]
            home_pmf = _pad_norm(poisson_binomial_pmf(home_xg_vals), max_goals + 1)
            away_pmf = _pad_norm(poisson_binomial_pmf(away_xg_vals), max_goals + 1)
            mat = scoreline_matrix_from_pmfs(home_pmf, away_pmf)
        elif approach == "double_poisson":
            home_pmf = double_poisson_pmf(float(row["home_xg"]), max_goals)
            away_pmf = double_poisson_pmf(float(row["away_xg"]), max_goals)
            mat = scoreline_matrix_from_pmfs(home_pmf, away_pmf)
        elif approach == "bivariate_poisson":
            l1 = max(float(row["home_xg"]) - float(lambda3), 1e-6)
            l2 = max(float(row["away_xg"]) - float(lambda3), 1e-6)
            mat = bivariate_poisson_matrix(l1, l2, float(lambda3), max_goals)
            home_pmf, away_pmf = _marginals_from_matrix(mat)
        elif approach == "diagonal_inflated_bivariate_poisson":
            l1 = max(float(row["home_xg"]) - float(lambda3), 1e-6)
            l2 = max(float(row["away_xg"]) - float(lambda3), 1e-6)
            mat = diagonal_inflated_bivariate_poisson_matrix(
                l1, l2, float(lambda3), float(p), float(theta), max_goals)
            home_pmf, away_pmf = _marginals_from_matrix(mat)
        elif approach == "dixon_coles":
            mat = dixon_coles_matrix(float(row["home_xg"]), float(row["away_xg"]),
                                     float(rho), max_goals)
            home_pmf, away_pmf = _marginals_from_matrix(mat)
        else:
            raise ValueError(f"Unknown approach: {approach}")

        rec = {c: row.get(c) for c in _META_SCORE_COLS}
        rec["home_pmf"]          = json.dumps(home_pmf.tolist())
        rec["away_pmf"]          = json.dumps(away_pmf.tolist())
        rec["scoreline_matrix"]  = json.dumps(mat.tolist())
        rec["approach"] = approach
        rows.append(rec)
    return pd.DataFrame(rows)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_probabilities(
    df: pd.DataFrame,
    output_dir: Path,
    competition_slug: str,
    approach: str,
    target: str,
) -> Path:
    """
    Save a predictions DataFrame to
    ``output_dir/<competition_slug>_<approach>_<target>.parquet``.

    Parameters
    ----------
    df : pd.DataFrame
        Predictions DataFrame (outcomes or scores schema).
    output_dir : Path
        Destination directory (created if absent).
    competition_slug : str
        Filename-safe competition identifier (e.g. ``wc2022``).
    approach : str
        Prediction approach (e.g. ``poisson_binomial``).
    target : str
        Prediction target: ``'outcomes'`` or ``'scores'``.

    Returns
    -------
    Path of the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(competition_slug)
    fname = f"{slug}_{_slug(approach)}_{_slug(target)}.parquet"
    path  = output_dir / fname
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows → {path}")
    return path


def load_probabilities(
    path: Path,
    parse_distributions: bool = False,
) -> pd.DataFrame:
    """
    Load a predictions parquet file.

    Parameters
    ----------
    path : Path
        Path to a ``*_outcomes.parquet`` or ``*_scores.parquet`` file.
    parse_distributions : bool
        If True and the file is a scores file, parse ``home_pmf``,
        ``away_pmf``, ``scoreline_matrix`` from JSON strings to Python objects.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_parquet(path)
    if parse_distributions:
        for col in ["home_pmf", "away_pmf", "scoreline_matrix"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: np.array(json.loads(v)) if isinstance(v, str) else v
                )
    return df


# ── Internal ──────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
