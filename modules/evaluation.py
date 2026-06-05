"""
evaluation.py
=============
Module C: Evaluate match-outcome and exact-score probability predictions.

Public API
----------
Outcome evaluation
    calculate_brier_score(df, prob_cols, outcome_col)
    ranked_probability_score(p_home, p_draw, p_away, outcome)
    mean_rps(df, prob_cols, outcome_col)
    expected_calibration_error(df, prob_cols, outcome_col, n_bins)
    compute_scoring_rules(models, outcome_col)           → summary DataFrame
    bootstrap_outcome_metrics(models, n_bootstrap, ci_level, rng_seed) → CI DataFrame
    plot_calibration_curves(models, n_bins, title, save_path) → Figure
    plot_scoring_comparison(scores_df, title, save_path)      → Figure
    plot_per_match_brier(df_a, df_b, name_a, name_b, save_path) → Figure

Score-distribution evaluation
    tvd(p, q)
    kl_divergence(p, q, eps)
    aggregate_score_distributions(scores_df, max_goals, max_total) → dict
    compute_empirical_distributions(match_df, max_goals, max_total) → dict
    compute_gof_metrics(empirical, theoretical_dict, max_sc)       → DataFrame
    bootstrap_gof_metrics(match_df, scores_dict, ...)              → CI DataFrame
    plot_goals_per_team(empirical, theoretical_dict, save_path)    → Figure
    plot_total_goals(empirical, theoretical_dict, save_path)       → Figure
    plot_scoreline_heatmaps(empirical_matrix, theoretical_dict, display, save_path)      → Figure
    plot_scoreline_diff_heatmaps(empirical_matrix, theoretical_dict, display, save_path) → Figure
    plot_top_scorelines(empirical_matrix, theoretical_dict, top_n, save_path) → Figure
    plot_top_scorelines_with_markers(empirical_matrix, theoretical_dict, top_n, save_path) → Figure
    plot_distribution_markers(empirical, theoretical_dict, ...)   → Figure
    plot_gof_summary(gof_df, save_path)                           → Figure

Input conventions
-----------------
*Outcome* DataFrames must have columns:
    actual_outcome  (str: 'home_win' | 'draw' | 'away_win')
    p_home_win, p_draw, p_away_win  (float probabilities)

*Score* DataFrames must have columns:
    home_pmf, away_pmf, scoreline_matrix  (JSON strings or list/ndarray)
    (as produced by probability_generation.save_probabilities)

All plotting functions return the matplotlib Figure and optionally save it
to ``save_path`` when provided.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

_OUTCOME_ORDER = ["home_win", "draw", "away_win"]
_PROB_COLS     = ["p_home_win", "p_draw", "p_away_win"]
_COLORS        = {
    "empirical":                             "#546E7A",
    "poisson_binomial":                      "#66BB6A",
    "double_poisson":                        "#42A5F5",
    "bivariate_poisson":                     "#EF9A9A",  # kept for backward compat
    "diagonal_inflated_bivariate_poisson":   "#EF5350",
    "dixon_coles":                           "#AB47BC",
    "betting_odds":                          "#FFA726",
    "default":                               "#8D6E63",
}
_CMAPS = ["Greens", "Blues", "Reds", "Purples", "Oranges", "BuGn", "YlOrBr"]


def _model_color(name: str) -> str:
    return _COLORS.get(name, _COLORS["default"])


def _save(fig: plt.Figure, save_path: Optional[Path]) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


# ── Outcome evaluation ────────────────────────────────────────────────────────

def calculate_brier_score(
    df: pd.DataFrame,
    prob_cols: list[str] = _PROB_COLS,
    outcome_col: str = "actual_outcome",
) -> float:
    """
    Multi-class Brier Score: sum of squared differences over all three outcome classes.

    Parameters
    ----------
    df : pd.DataFrame
        Predictions DataFrame with actual outcomes and probability columns.
    prob_cols : list of str
        Column names for [p_home_win, p_draw, p_away_win].
    outcome_col : str
        Column containing actual outcome strings.

    Returns
    -------
    float — mean multi-class Brier Score (lower is better).
    """
    total = 0.0
    for outcome, col in zip(_OUTCOME_ORDER, prob_cols):
        y_true = (df[outcome_col] == outcome).astype(float)
        y_pred = df[col].astype(float)
        total += float(np.mean((y_pred - y_true) ** 2))
    return total


def ranked_probability_score(
    p_home: float,
    p_draw: float,
    p_away: float,
    outcome: str,
) -> float:
    """
    RPS for a single three-outcome football match.

    Outcomes are ordered (home_win < draw < away_win).
    Lower is better; uniform 1/3 baseline → 1/3.
    """
    F1 = p_home
    F2 = p_home + p_draw
    if outcome == "home_win":
        G1, G2 = 1.0, 1.0
    elif outcome == "draw":
        G1, G2 = 0.0, 1.0
    else:
        G1, G2 = 0.0, 0.0
    return 0.5 * ((F1 - G1) ** 2 + (F2 - G2) ** 2)


def mean_rps(
    df: pd.DataFrame,
    prob_cols: list[str] = _PROB_COLS,
    outcome_col: str = "actual_outcome",
) -> tuple[float, pd.Series]:
    """
    Mean Ranked Probability Score over all matches.

    Returns
    -------
    (mean_rps, per_match_series)
    """
    p_home_col, p_draw_col, p_away_col = prob_cols
    series = df.apply(
        lambda row: ranked_probability_score(
            float(row[p_home_col]),
            float(row[p_draw_col]),
            float(row[p_away_col]),
            row[outcome_col],
        ),
        axis=1,
    )
    return float(series.mean()), series


def expected_calibration_error(
    df: pd.DataFrame,
    prob_cols: list[str] = _PROB_COLS,
    outcome_col: str = "actual_outcome",
    n_bins: int = 10,
) -> tuple[float, list[float]]:
    """
    Expected Calibration Error (ECE), macro-averaged over three outcome classes.

    ECE = Σ_m |B_m|/n · |mean_conf(B_m) − mean_acc(B_m)|

    Returns
    -------
    (macro_ece, [ece_home_win, ece_draw, ece_away_win])
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_list: list[float] = []
    for outcome, col in zip(_OUTCOME_ORDER, prob_cols):
        y_true = (df[outcome_col] == outcome).astype(float).values
        y_pred = df[col].astype(float).values
        bin_idx = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins - 1)
        n = len(y_true)
        ece = 0.0
        for m in range(n_bins):
            mask = bin_idx == m
            if not mask.any():
                continue
            acc  = y_true[mask].mean()
            conf = y_pred[mask].mean()
            ece += (mask.sum() / n) * abs(acc - conf)
        ece_list.append(float(ece))
    return float(np.mean(ece_list)), ece_list


def compute_scoring_rules(
    models: dict[str, pd.DataFrame],
    outcome_col: str = "actual_outcome",
    prob_cols: list[str] = _PROB_COLS,
) -> pd.DataFrame:
    """
    Compute Brier, RPS, and ECE for each approach.

    Parameters
    ----------
    models : dict {approach_name: predictions_df}
        Each DataFrame must contain ``prob_cols`` and ``outcome_col``.
    outcome_col : str
    prob_cols : list of str

    Returns
    -------
    pd.DataFrame with columns [Approach, Brier, RPS, ECE, N]
    """
    rows = []
    for name, df in models.items():
        brier         = calculate_brier_score(df, prob_cols, outcome_col)
        rps, _        = mean_rps(df, prob_cols, outcome_col)
        ece, ece_list = expected_calibration_error(df, prob_cols, outcome_col)
        rows.append({
            "Approach": name,
            "N":        len(df),
            "Brier":    round(brier, 4),
            "RPS":      round(rps,   4),
            "ECE":      round(ece,   4),
            "ECE_home": round(ece_list[0], 4),
            "ECE_draw": round(ece_list[1], 4),
            "ECE_away": round(ece_list[2], 4),
        })
    return pd.DataFrame(rows)


def bootstrap_outcome_metrics(
    models: dict[str, pd.DataFrame],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng_seed: int = 42,
    prob_cols: list[str] = _PROB_COLS,
    outcome_col: str = "actual_outcome",
) -> pd.DataFrame:
    """
    Percentile-bootstrap confidence intervals for Brier and RPS.

    For each approach the match-level predictions are resampled with replacement
    ``n_bootstrap`` times; point estimates are recomputed on every resample.

    Parameters
    ----------
    models : dict {approach_name: predictions_df}
    n_bootstrap : int
    ci_level : float — nominal coverage, e.g. 0.95 for 95 % CI
    rng_seed : int

    Returns
    -------
    pd.DataFrame with columns:
        Approach, Brier_lower, Brier_upper, Brier_se,
                  RPS_lower,   RPS_upper,   RPS_se
    """
    rng   = np.random.default_rng(rng_seed)
    alpha = (1 - ci_level) / 2
    rows  = []
    for name, df in models.items():
        df_r       = df.reset_index(drop=True)
        n          = len(df_r)
        brier_boot = np.empty(n_bootstrap)
        rps_boot   = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx            = rng.integers(0, n, size=n)
            sample         = df_r.iloc[idx]
            brier_boot[b]  = calculate_brier_score(sample, prob_cols, outcome_col)
            rps_boot[b], _ = mean_rps(sample, prob_cols, outcome_col)
        rows.append({
            "Approach":    name,
            "Brier_lower": float(np.quantile(brier_boot, alpha)),
            "Brier_upper": float(np.quantile(brier_boot, 1 - alpha)),
            "Brier_se":    float(brier_boot.std()),
            "RPS_lower":   float(np.quantile(rps_boot, alpha)),
            "RPS_upper":   float(np.quantile(rps_boot, 1 - alpha)),
            "RPS_se":      float(rps_boot.std()),
        })
    return pd.DataFrame(rows)


def plot_calibration_curves(
    models: dict[str, pd.DataFrame],
    n_bins: int = 10,
    title: str = "",
    save_path: Optional[Path] = None,
    outcome_col: str = "actual_outcome",
    prob_cols: list[str] = _PROB_COLS,
) -> plt.Figure:
    """
    Reliability diagrams (one subplot per outcome class) comparing all approaches.

    Parameters
    ----------
    models : dict {approach_name: df}
    n_bins : int
    title : str — figure suptitle suffix
    save_path : Path, optional
    """
    labels = ["Home Win", "Draw", "Away Win"]
    bins   = np.linspace(0.0, 1.0, n_bins + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, outcome, label, prob_col in zip(axes, _OUTCOME_ORDER, labels, prob_cols):
        for name, df in models.items():
            y_true  = (df[outcome_col] == outcome).astype(float).values
            y_pred  = df[prob_col].astype(float).values
            bin_idx = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins - 1)

            bin_conf, bin_acc = [], []
            for m in range(n_bins):
                mask = bin_idx == m
                if not mask.any():
                    continue
                bin_conf.append(y_pred[mask].mean())
                bin_acc.append(y_true[mask].mean())

            color = _model_color(name)
            ax.plot(bin_conf, bin_acc, color=color, linewidth=1.5, alpha=0.85)
            ax.scatter(bin_conf, bin_acc, color=color, s=50, zorder=5, label=name)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.suptitle(
        f"Calibration Curves{' — ' + title if title else ''}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_scoring_comparison(
    scores_df: pd.DataFrame,
    title: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart comparing Brier, RPS, and ECE across approaches.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of ``compute_scoring_rules``.
    """
    approaches = scores_df["Approach"].tolist()
    colors     = [_model_color(a) for a in approaches]

    metrics = [
        ("Brier", 0.6667, "Uniform baseline (0.6667)"),
        ("RPS",   0.3333, "Uniform baseline (0.3333)"),
        ("ECE",   None,   None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (metric, baseline, bl_label) in zip(axes, metrics):
        vals = scores_df[metric].tolist()
        bars = ax.bar(approaches, vals, color=colors, alpha=0.85, edgecolor="black")
        if baseline is not None:
            ax.axhline(y=baseline, color="red", linestyle=":", linewidth=2, label=bl_label)
            ax.legend(fontsize=8)
        ax.set_title(f"{metric}\n(lower is better)", fontsize=11)
        ax.set_ylabel(metric)
        y_max = max(vals) * 1.25
        ax.set_ylim(0, y_max)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.tick_params(axis="x", labelrotation=15)

    plt.suptitle(
        f"Scoring Rule Comparison{' — ' + title if title else ''}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_per_match_brier(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    save_path: Optional[Path] = None,
    outcome_col: str = "actual_outcome",
    prob_cols: list[str] = _PROB_COLS,
) -> plt.Figure:
    """
    Three-panel per-match Brier comparison between two approaches.

    Panel 1: scatter plot of per-match Brier scores.
    Panel 2: histogram of Brier difference (A − B).
    Panel 3: bar chart of mean scores.

    Both DataFrames must be aligned (same match order / same index).
    """
    def _per_match_brier(df: pd.DataFrame) -> pd.Series:
        total = pd.Series(0.0, index=df.index)
        for outcome, col in zip(_OUTCOME_ORDER, prob_cols):
            y_true = (df[outcome_col] == outcome).astype(float)
            y_pred = df[col].astype(float)
            total += (y_pred - y_true) ** 2
        return total

    brier_a = _per_match_brier(df_a).values
    brier_b = _per_match_brier(df_b).values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: scatter
    ax = axes[0]
    ax.scatter(brier_b, brier_a, alpha=0.6, s=60, color=_model_color(name_a))
    lim = max(brier_a.max(), brier_b.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", label="Equal performance")
    a_better = int((brier_a < brier_b).sum())
    b_better = int((brier_a > brier_b).sum())
    ax.text(
        0.05, 0.95,
        f"{name_a} better: {a_better}\n{name_b} better: {b_better}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_xlabel(f"{name_b} Brier Score")
    ax.set_ylabel(f"{name_a} Brier Score")
    ax.set_title("Per-Match Brier Score")
    ax.legend(fontsize=8)

    # Panel 2: histogram of difference
    ax = axes[1]
    diff = brier_b - brier_a  # positive = A is better
    ax.hist(diff, bins=20, edgecolor="black", alpha=0.7, color=_model_color(name_a))
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No difference")
    ax.axvline(x=diff.mean(), color="green", linewidth=2,
               label=f"Mean: {diff.mean():.3f}")
    ax.set_xlabel(f"Brier Difference ({name_b} − {name_a})")
    ax.set_ylabel("Number of Matches")
    ax.set_title(f"Improvement of {name_a} over {name_b}")
    ax.legend(fontsize=8)

    # Panel 3: mean bar
    ax = axes[2]
    means = [brier_a.mean(), brier_b.mean()]
    colors = [_model_color(name_a), _model_color(name_b)]
    bars = ax.bar([name_a, name_b], means, color=colors, alpha=0.85, edgecolor="black")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean Brier Score")
    ax.set_title("Mean Brier Score Comparison")

    plt.suptitle(
        f"Per-Match Brier Comparison: {name_a} vs {name_b}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ── Score-distribution evaluation ─────────────────────────────────────────────

def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total Variation Distance: 0.5 · Σ|p_i − q_i|.  Range [0, 1]."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(0.5 * np.abs(p - q).sum())


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """KL divergence D(p ‖ q).  p is empirical, q is theoretical."""
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _parse_json_col(series: pd.Series) -> list:
    """Parse a column of JSON strings or lists/arrays to Python lists."""
    out = []
    for v in series:
        if isinstance(v, str):
            out.append(json.loads(v))
        elif isinstance(v, np.ndarray):
            out.append(v.tolist())
        else:
            out.append(list(v))
    return out


def aggregate_score_distributions(
    scores_df: pd.DataFrame,
    max_goals: int = 10,
    max_total: int = 12,
) -> dict[str, np.ndarray]:
    """
    Compute the average predicted goal distributions over all matches.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Scores DataFrame as produced by probability_generation generators.
        Must contain ``home_pmf``, ``away_pmf``, ``scoreline_matrix`` columns
        (JSON strings or list/array values).
    max_goals : int
        Goals cap for per-team PMF (length = max_goals + 1).
    max_total : int
        Total goals cap (length = max_total + 1).

    Returns
    -------
    dict with keys:
        'home'   – avg home-goals PMF, shape (max_goals+1,)
        'away'   – avg away-goals PMF, shape (max_goals+1,)
        'total'  – avg total-goals PMF, shape (max_total+1,)
        'matrix' – avg scoreline matrix, shape (max_goals+1, max_goals+1)
    """
    n = max_goals + 1
    home_pmfs  = _parse_json_col(scores_df["home_pmf"])
    away_pmfs  = _parse_json_col(scores_df["away_pmf"])
    sc_matrices = _parse_json_col(scores_df["scoreline_matrix"])

    avg_home = np.zeros(n)
    avg_away = np.zeros(n)
    avg_mat  = np.zeros((n, n))
    avg_tot  = np.zeros(max_total + 1)

    for h_pmf, a_pmf, mat in zip(home_pmfs, away_pmfs, sc_matrices):
        h = np.array(h_pmf, float)
        a = np.array(a_pmf, float)
        m = np.array(mat,   float)

        # pad / truncate to n
        h = np.pad(h, (0, max(0, n - len(h))))[:n]
        a = np.pad(a, (0, max(0, n - len(a))))[:n]
        m = np.pad(m, ((0, max(0, n - m.shape[0])), (0, max(0, n - m.shape[1]))))[:n, :n]

        # renormalise
        if h.sum() > 0:
            h /= h.sum()
        if a.sum() > 0:
            a /= a.sum()
        if m.sum() > 0:
            m /= m.sum()

        # Total goals = anti-diagonal sums of the joint matrix.
        # Convolving the marginals would only be correct under independence;
        # for Bivariate Poisson and Dixon–Coles the joint induces dependence
        # (DC's τ even preserves the marginals exactly, so the marginal
        # convolution of DP and DC are mathematically identical and the
        # dependence-induced redistribution would be invisible).
        n_mat = m.shape[0]
        tot = np.zeros(2 * n_mat - 1)
        for hi in range(n_mat):
            tot[hi:hi + n_mat] += m[hi, :]
        tot = tot[: max_total + 1]
        if tot.sum() > 0:
            tot /= tot.sum()

        avg_home += h
        avg_away += a
        avg_mat  += m
        avg_tot  += tot

    k = len(scores_df)
    return {
        "home":   avg_home  / k,
        "away":   avg_away  / k,
        "total":  avg_tot   / k,
        "matrix": avg_mat   / k,
    }


def compute_empirical_distributions(
    match_df: pd.DataFrame,
    max_goals: int = 10,
    max_total: int = 12,
) -> dict[str, np.ndarray]:
    """
    Compute empirical goal distributions from observed match results.

    Parameters
    ----------
    match_df : pd.DataFrame
        Must contain ``home_goals`` and ``away_goals`` columns.

    Returns
    -------
    dict with keys 'home', 'away', 'total', 'matrix'  (same shape conventions
    as :func:`aggregate_score_distributions`).
    """
    n   = max_goals + 1
    hg  = match_df["home_goals"].clip(upper=max_goals).astype(int)
    ag  = match_df["away_goals"].clip(upper=max_goals).astype(int)
    tg  = (match_df["home_goals"] + match_df["away_goals"]).clip(upper=max_total).astype(int)

    def _pmf(vals, length):
        counts = np.bincount(vals, minlength=length)
        return counts.astype(float) / counts.sum()

    emp_home  = _pmf(hg, n)
    emp_away  = _pmf(ag, n)
    emp_total = _pmf(tg, max_total + 1)

    emp_matrix = np.zeros((n, n))
    for h, a in zip(hg, ag):
        emp_matrix[h, a] += 1
    emp_matrix /= len(match_df)

    return {
        "home":   emp_home,
        "away":   emp_away,
        "total":  emp_total,
        "matrix": emp_matrix,
    }


def compute_gof_metrics(
    empirical: dict[str, np.ndarray],
    theoretical_dict: dict[str, dict[str, np.ndarray]],
    max_sc: int = 7,
) -> pd.DataFrame:
    """
    Compute TVD and KL divergence for each theoretical approach.

    Parameters
    ----------
    empirical : dict
        Output of :func:`compute_empirical_distributions`.
    theoretical_dict : dict {approach_name: dist_dict}
        Each value is the output of :func:`aggregate_score_distributions`.
    max_sc : int
        Scoreline matrix is cropped to [0..max_sc] × [0..max_sc] before
        flattening to compute scoreline TVD/KL.

    Returns
    -------
    pd.DataFrame with columns:
        Distribution, Approach, TVD, KL
    """
    rows = []
    for dist_name, (emp_key, emp_arr) in [
        ("Home goals",   ("home",   empirical["home"])),
        ("Away goals",   ("away",   empirical["away"])),
        ("Total goals",  ("total",  empirical["total"])),
        ("Exact scorelines",
         ("matrix", empirical["matrix"][:max_sc + 1, :max_sc + 1])),
    ]:
        e = emp_arr.flatten()
        if e.sum() > 0:
            e = e / e.sum()
        for name, theo in theoretical_dict.items():
            if dist_name == "Exact scorelines":
                t = theo["matrix"][:max_sc + 1, :max_sc + 1].flatten()
            else:
                t = theo[emp_key].flatten()
            if t.sum() > 0:
                t = t / t.sum()
            rows.append({
                "Distribution": dist_name,
                "Approach": name,
                "TVD": round(tvd(e, t),          4),
                "KL":  round(kl_divergence(e, t), 4),
            })
    return pd.DataFrame(rows)


def bootstrap_gof_metrics(
    match_df: pd.DataFrame,
    scores_dict: dict[str, pd.DataFrame],
    max_goals: int = 10,
    max_total: int = 12,
    max_sc: int = 7,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Percentile-bootstrap confidence intervals for TVD and KL divergence.

    Matches are aligned by *match_id* (when the column is present in both
    ``match_df`` and every entry of ``scores_dict``); otherwise datasets are
    aligned positionally on the common prefix.

    Each bootstrap iteration resamples matches jointly so the empirical and
    theoretical distributions are always derived from the same set of matches.
    The inner loop is vectorised with pre-stacked numpy arrays (each bootstrap
    replicate is a single ``ndarray[idx].mean(0)`` call) so 1 000 iterations
    over ~1 800 matches completes in a few seconds.

    Parameters
    ----------
    match_df : pd.DataFrame
        Match results — must contain ``home_goals`` and ``away_goals``.
    scores_dict : dict {approach_name: scores_df}
        Per-match score predictions — must contain ``home_pmf``,
        ``away_pmf``, ``scoreline_matrix``.
    n_bootstrap : int
    ci_level : float
    rng_seed : int

    Returns
    -------
    pd.DataFrame with columns:
        Distribution, Approach,
        TVD_lower, TVD_upper, TVD_se,
        KL_lower,  KL_upper,  KL_se
    """
    rng   = np.random.default_rng(rng_seed)
    alpha = (1 - ci_level) / 2

    # ── Align datasets on match_id ─────────────────────────────────────────
    has_id = (
        "match_id" in match_df.columns
        and all("match_id" in df.columns for df in scores_dict.values())
    )
    if has_id:
        all_ids   = sorted(
            set(match_df["match_id"].unique()).intersection(
                *[set(df["match_id"].unique()) for df in scores_dict.values()]
            )
        )
        match_sub  = match_df.set_index("match_id").loc[all_ids].reset_index()
        scores_sub = {
            a: df.set_index("match_id").loc[all_ids].reset_index()
            for a, df in scores_dict.items()
        }
    else:
        n_common   = min(len(match_df), *(len(df) for df in scores_dict.values()))
        match_sub  = match_df.iloc[:n_common].reset_index(drop=True)
        scores_sub = {
            a: df.iloc[:n_common].reset_index(drop=True)
            for a, df in scores_dict.items()
        }

    n  = len(match_sub)
    ng = max_goals + 1

    # ── Pre-stack empirical one-hot arrays ─────────────────────────────────
    hg = match_sub["home_goals"].clip(upper=max_goals).astype(int).values
    ag = match_sub["away_goals"].clip(upper=max_goals).astype(int).values
    tg = np.clip(hg + ag, 0, max_total).astype(int)

    e_hg_oh = np.zeros((n, ng),            dtype=float)
    e_ag_oh = np.zeros((n, ng),            dtype=float)
    e_tg_oh = np.zeros((n, max_total + 1), dtype=float)
    sc_dim  = (max_sc + 1) ** 2
    e_sc_oh = np.zeros((n, sc_dim),        dtype=float)

    for i in range(n):
        e_hg_oh[i, hg[i]] = 1.0
        e_ag_oh[i, ag[i]] = 1.0
        e_tg_oh[i, tg[i]] = 1.0
        if hg[i] <= max_sc and ag[i] <= max_sc:
            e_sc_oh[i, hg[i] * (max_sc + 1) + ag[i]] = 1.0

    # ── Pre-stack theoretical PMF arrays ──────────────────────────────────
    theo_stacked: dict[str, dict[str, np.ndarray]] = {}
    for a, sdf in scores_sub.items():
        home_pmfs   = _parse_json_col(sdf["home_pmf"])
        away_pmfs   = _parse_json_col(sdf["away_pmf"])
        sc_matrices = _parse_json_col(sdf["scoreline_matrix"])

        t_h = np.zeros((n, ng))
        t_a = np.zeros((n, ng))
        t_t = np.zeros((n, max_total + 1))
        t_s = np.zeros((n, sc_dim))

        for i, (hp, ap, mat) in enumerate(zip(home_pmfs, away_pmfs, sc_matrices)):
            h = np.array(hp,  float)
            av = np.array(ap, float)
            m = np.array(mat, float)

            # pad / truncate to ng × ng
            h  = np.pad(h,  (0, max(0, ng - len(h))))[:ng]
            av = np.pad(av, (0, max(0, ng - len(av))))[:ng]
            m  = np.pad(
                m,
                ((0, max(0, ng - m.shape[0])), (0, max(0, ng - m.shape[1]))),
            )[:ng, :ng]

            if h.sum()  > 0: h  /= h.sum()
            if av.sum() > 0: av /= av.sum()
            if m.sum()  > 0: m  /= m.sum()

            # total from anti-diagonal sums of the joint matrix
            tot = np.zeros(2 * ng - 1)
            for hi in range(ng):
                tot[hi: hi + ng] += m[hi]
            tot = tot[: max_total + 1]
            if tot.sum() > 0: tot /= tot.sum()

            mc = m[: max_sc + 1, : max_sc + 1]
            if mc.sum() > 0: mc /= mc.sum()

            t_h[i] = h
            t_a[i] = av
            t_t[i] = tot
            t_s[i] = mc.flatten()

        theo_stacked[a] = {"home": t_h, "away": t_a, "total": t_t, "scoreline": t_s}

    # ── Bootstrap loop ─────────────────────────────────────────────────────
    dist_names = ["Home goals", "Away goals", "Total goals", "Exact scorelines"]
    tvd_boot = {d: {a: np.empty(n_bootstrap) for a in scores_sub} for d in dist_names}
    kl_boot  = {d: {a: np.empty(n_bootstrap) for a in scores_sub} for d in dist_names}

    def _norm(v: np.ndarray) -> np.ndarray:
        s = v.sum()
        return v / s if s > 0 else v

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)

        e_h = _norm(e_hg_oh[idx].mean(0))
        e_a = _norm(e_ag_oh[idx].mean(0))
        e_t = _norm(e_tg_oh[idx].mean(0))
        e_s = _norm(e_sc_oh[idx].mean(0))

        for a, ts in theo_stacked.items():
            t_h = _norm(ts["home"][idx].mean(0))
            t_a = _norm(ts["away"][idx].mean(0))
            t_t = _norm(ts["total"][idx].mean(0))
            t_s = _norm(ts["scoreline"][idx].mean(0))

            tvd_boot["Home goals"][a][b]       = tvd(e_h, t_h)
            kl_boot["Home goals"][a][b]        = kl_divergence(e_h, t_h)
            tvd_boot["Away goals"][a][b]       = tvd(e_a, t_a)
            kl_boot["Away goals"][a][b]        = kl_divergence(e_a, t_a)
            tvd_boot["Total goals"][a][b]      = tvd(e_t, t_t)
            kl_boot["Total goals"][a][b]       = kl_divergence(e_t, t_t)
            tvd_boot["Exact scorelines"][a][b] = tvd(e_s, t_s)
            kl_boot["Exact scorelines"][a][b]  = kl_divergence(e_s, t_s)

    # ── Collect results ────────────────────────────────────────────────────
    rows = []
    for dist_name in dist_names:
        for a in scores_sub:
            tb = tvd_boot[dist_name][a]
            kb = kl_boot[dist_name][a]
            rows.append({
                "Distribution": dist_name,
                "Approach":     a,
                "TVD_lower":    float(np.quantile(tb, alpha)),
                "TVD_upper":    float(np.quantile(tb, 1 - alpha)),
                "TVD_se":       float(tb.std()),
                "KL_lower":     float(np.quantile(kb, alpha)),
                "KL_upper":     float(np.quantile(kb, 1 - alpha)),
                "KL_se":        float(kb.std()),
            })
    return pd.DataFrame(rows)


# ── Score distribution plots ──────────────────────────────────────────────────

def plot_goals_per_team(
    empirical: dict[str, np.ndarray],
    theoretical_dict: dict[str, dict[str, np.ndarray]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart: home-team and away-team goal distributions.
    """
    n_models = len(theoretical_dict)
    bar_w    = 0.8 / (n_models + 1)
    goal_range = np.arange(len(empirical["home"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, side in zip(axes, ["home", "away"]):
        emp = empirical[side]
        offset = -(n_models / 2) * bar_w
        ax.bar(goal_range + offset, emp, bar_w,
               label="Empirical", color=_COLORS["empirical"],
               alpha=0.85, edgecolor="black")
        for i, (name, theo) in enumerate(theoretical_dict.items()):
            off = offset + (i + 1) * bar_w
            ax.bar(goal_range + off, theo[side], bar_w,
                   label=name, color=_model_color(name),
                   alpha=0.85, edgecolor="black")
        ax.set_xlabel("Goals")
        ax.set_ylabel("Probability")
        ax.set_title(f"{'Home' if side == 'home' else 'Away'} Team Goals")
        ax.set_xticks(goal_range)
        ax.legend(fontsize=8)
        # annotate TVD
        lines = ["TVD"]
        for name, theo in theoretical_dict.items():
            lines.append(f"{name}: {tvd(emp, theo[side]):.4f}")
        ax.text(0.98, 0.97, "\n".join(lines),
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.suptitle("Goal Distribution per Team — Empirical vs Theoretical",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_total_goals(
    empirical: dict[str, np.ndarray],
    theoretical_dict: dict[str, dict[str, np.ndarray]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart: total goals (home + away) distribution.
    """
    emp        = empirical["total"]
    total_range = np.arange(len(emp))
    n_models   = len(theoretical_dict)
    bar_w      = 0.8 / (n_models + 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    offset = -(n_models / 2) * bar_w
    ax.bar(total_range + offset, emp, bar_w,
           label="Empirical", color=_COLORS["empirical"],
           alpha=0.85, edgecolor="black")
    for i, (name, theo) in enumerate(theoretical_dict.items()):
        off = offset + (i + 1) * bar_w
        ax.bar(total_range + off, theo["total"], bar_w,
               label=name, color=_model_color(name),
               alpha=0.85, edgecolor="black")

    lines = ["TVD"]
    for name, theo in theoretical_dict.items():
        lines.append(f"{name}: {tvd(emp, theo['total']):.4f}")
    ax.text(0.98, 0.97, "\n".join(lines),
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Total Goals (Home + Away)")
    ax.set_ylabel("Probability")
    ax.set_title("Total Goals Distribution — Empirical vs Theoretical",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(total_range)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_scoreline_heatmaps(
    empirical_matrix: np.ndarray,
    theoretical_dict: dict[str, dict[str, np.ndarray]],
    display: int = 8,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Heatmaps of empirical and theoretical scoreline probability matrices (%).
    """
    n_panels = 1 + len(theoretical_dict)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    # empirical
    ax = axes[0]
    mat = empirical_matrix[:display, :display] * 100
    sns.heatmap(mat, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=range(display), yticklabels=range(display),
                ax=ax, cbar_kws={"label": "Probability (%)"})
    ax.set_xlabel("Away Goals")
    ax.set_ylabel("Home Goals")
    ax.set_title("Empirical", fontsize=12)

    for ax, (name, theo), cmap in zip(axes[1:], theoretical_dict.items(), _CMAPS):
        mat = theo["matrix"][:display, :display] * 100
        sns.heatmap(mat, annot=True, fmt=".1f", cmap=cmap,
                    xticklabels=range(display), yticklabels=range(display),
                    ax=ax, cbar_kws={"label": "Probability (%)"})
        ax.set_xlabel("Away Goals")
        ax.set_ylabel("Home Goals")
        ax.set_title(name, fontsize=12)

    plt.suptitle("Exact Scoreline Probability Matrices (%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_scoreline_diff_heatmaps(
    empirical_matrix: np.ndarray,
    theoretical_dict: dict[str, dict[str, np.ndarray]],
    display: int = 8,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Difference heatmaps: theoretical − empirical (in percentage points).
    """
    n = len(theoretical_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    emp = empirical_matrix[:display, :display]
    for ax, (name, theo) in zip(axes, theoretical_dict.items()):
        diff = (theo["matrix"][:display, :display] - emp) * 100
        vabs = max(abs(diff.min()), abs(diff.max()), 0.1)
        sns.heatmap(diff, annot=True, fmt=".1f", cmap="RdBu_r",
                    center=0, vmin=-vabs, vmax=vabs,
                    xticklabels=range(display), yticklabels=range(display),
                    ax=ax, cbar_kws={"label": "Diff (pp)"})
        ax.set_xlabel("Away Goals")
        ax.set_ylabel("Home Goals")
        ax.set_title(f"{name} − Empirical", fontsize=12)

    plt.suptitle("Scoreline Prediction Error (Theoretical − Empirical, pp)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_top_scorelines(
    empirical_matrix: np.ndarray,
    theoretical_dict: dict[str, dict[str, np.ndarray]],
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart for the top-N most frequent observed scorelines.
    """
    n_goals = empirical_matrix.shape[0]
    records = []
    for h in range(n_goals):
        for a in range(n_goals):
            rec = {"Scoreline": f"{h}:{a}", "Empirical": empirical_matrix[h, a]}
            for name, theo in theoretical_dict.items():
                rec[name] = theo["matrix"][h, a] if h < theo["matrix"].shape[0] and a < theo["matrix"].shape[1] else 0.0
            records.append(rec)

    sl_df = pd.DataFrame(records).sort_values("Empirical", ascending=False).head(top_n)

    n_models = len(theoretical_dict)
    bar_w    = 0.8 / (n_models + 1)
    x        = np.arange(top_n)

    fig, ax = plt.subplots(figsize=(16, 6))
    offset = -(n_models / 2) * bar_w
    ax.bar(x + offset, sl_df["Empirical"], bar_w,
           label="Empirical", color=_COLORS["empirical"],
           alpha=0.85, edgecolor="black")
    for i, (name, theo) in enumerate(theoretical_dict.items()):
        off = offset + (i + 1) * bar_w
        ax.bar(x + off, sl_df[name], bar_w,
               label=name, color=_model_color(name),
               alpha=0.85, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(sl_df["Scoreline"], rotation=45, ha="right")
    ax.set_xlabel("Scoreline (Home:Away)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Top {top_n} Scorelines — Empirical vs Theoretical",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_top_scorelines_with_markers(
    empirical_matrix: np.ndarray,
    theoretical_dict: dict[str, dict[str, np.ndarray]],
    top_n: int = 20,
    save_path: Optional[Path] = None,
    jitter: float = 0.12,
) -> plt.Figure:
    """
    Top-N scoreline chart with the empirical distribution as bars and each
    theoretical model's predicted probability overlaid as a scatter marker
    on the same x position.

    ``jitter`` spreads same-x markers horizontally so closely-clustered
    models can be visually disentangled.
    """
    n_goals = empirical_matrix.shape[0]
    records = []
    for h in range(n_goals):
        for a in range(n_goals):
            rec = {"Scoreline": f"{h}:{a}", "Empirical": empirical_matrix[h, a]}
            for name, theo in theoretical_dict.items():
                m = theo["matrix"]
                rec[name] = m[h, a] if h < m.shape[0] and a < m.shape[1] else 0.0
            records.append(rec)

    sl_df = (
        pd.DataFrame(records)
        .sort_values("Empirical", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    x = np.arange(len(sl_df))
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(
        x, sl_df["Empirical"], width=0.7,
        label="Empirical", color=_COLORS["empirical"],
        alpha=0.85, edgecolor="black", zorder=1,
    )

    n_models = max(len(theoretical_dict), 1)
    offsets = np.linspace(-jitter, jitter, n_models) if n_models > 1 else np.array([0.0])
    markers = ["o", "D", "s", "^", "v", "P", "X"]
    for i, (name, _) in enumerate(theoretical_dict.items()):
        ax.scatter(
            x + offsets[i], sl_df[name],
            label=name,
            color=_model_color(name),
            marker=markers[i % len(markers)],
            s=80, edgecolor="black", linewidth=0.8,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sl_df["Scoreline"], rotation=45, ha="right")
    ax.set_xlabel("Scoreline (Home:Away)")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Top {top_n} Scorelines — Empirical (bars) vs Theoretical (markers)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_distribution_markers(
    empirical: np.ndarray,
    theoretical_dict: dict[str, np.ndarray],
    x_labels: Optional[list[str]] = None,
    jitter: float = 0.12,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "",
    ylabel: str = "Probability",
    rotate: int = 0,
    save_path: Optional[Path] = None,
    rng_seed: int = 0,
) -> plt.Figure:
    """
    Bar chart of an empirical PMF with each theoretical model overlaid as
    horizontally-jittered scatter markers on the same x-positions.

    Useful when several models predict similar values: jitter spreads the
    points so they remain distinguishable.

    Parameters
    ----------
    empirical : np.ndarray
        Empirical probability mass values.
    theoretical_dict : dict {approach_name: np.ndarray}
        One PMF per approach, all sharing the same support as ``empirical``.
    x_labels : list of str, optional
        Tick labels.  Defaults to integer indices.
    jitter : float
        Maximum horizontal offset (in tick-units) applied to scatter points.
        Each model is offset by an evenly-spaced value in [-jitter, +jitter].
    ax : matplotlib Axes, optional
        Draw onto an existing axes (no new figure created).
    """
    n = len(empirical)
    x = np.arange(n)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * n + 4), 5))
    else:
        fig = ax.figure

    ax.bar(x, empirical, width=0.7,
           label="Empirical", color=_COLORS["empirical"],
           alpha=0.85, edgecolor="black", zorder=1)

    n_models = max(len(theoretical_dict), 1)
    if n_models == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-jitter, jitter, n_models)

    markers = ["o", "D", "s", "^", "v", "P", "X"]
    for i, (name, pmf) in enumerate(theoretical_dict.items()):
        ax.scatter(x + offsets[i], pmf,
                   label=name,
                   color=_model_color(name),
                   marker=markers[i % len(markers)],
                   s=55, edgecolor="black", linewidth=0.7,
                   zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels if x_labels is not None else [str(int(v)) for v in x],
                       rotation=rotate, ha="right" if rotate else "center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=9)

    if own_fig:
        plt.tight_layout()
        _save(fig, save_path)
    return fig


def plot_gof_summary(
    gof_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart summarising TVD and KL across distributions and approaches.

    Parameters
    ----------
    gof_df : pd.DataFrame
        Output of :func:`compute_gof_metrics`.
    """
    distributions = gof_df["Distribution"].unique().tolist()
    approaches    = gof_df["Approach"].unique().tolist()
    x             = np.arange(len(distributions))
    n_models      = len(approaches)
    bar_w         = 0.8 / n_models

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["TVD", "KL"]):
        for i, approach in enumerate(approaches):
            sub  = gof_df[gof_df["Approach"] == approach]
            vals = [float(sub.loc[sub["Distribution"] == d, metric].iloc[0])
                    if d in sub["Distribution"].values else 0.0
                    for d in distributions]
            off  = (i - n_models / 2 + 0.5) * bar_w
            bars = ax.bar(x + off, vals, bar_w,
                          label=approach, color=_model_color(approach),
                          alpha=0.85, edgecolor="black")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(distributions, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}\n(lower = better fit)", fontsize=11)
        ax.legend(fontsize=8)

    plt.suptitle("Model Fit to Empirical Distributions",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig
