"""
data_loading.py
===============
Module A: Load, process and save competition data.

Public API
----------
get_available_competitions()
load_statsbomb_competition(competition_id, season_id, ...)
process_shots_to_match_summary(shots_df, matches_df)
load_football_data_odds(csv_path, competition_name, ...)
merge_odds_with_matches(match_summary, odds_df)
save_competition_data(match_summary, output_dir, competition_name)
load_competition_data(path)
TEAM_NAME_MAP_DEFAULT

Saved file schema (input_data/*.parquet)
-----------------------------------------
match_id, match_date, competition_name, season_name,
competition_stage (nullable), match_week (nullable),
home_team, away_team, home_goals, away_goals,
home_xg, away_xg, home_shots, away_shots,
home_xg_values (JSON str), away_xg_values (JSON str),
betting_p_home (nullable), betting_p_draw (nullable), betting_p_away (nullable)
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ── Team-name mapping: football-data.co.uk → StatsBomb ───────────────────────

TEAM_NAME_MAP_DEFAULT: dict[str, str] = {
    # Premier League
    "Bournemouth":   "AFC Bournemouth",
    "Leicester":     "Leicester City",
    "Man City":      "Manchester City",
    "Man United":    "Manchester United",
    "Newcastle":     "Newcastle United",
    "Norwich":       "Norwich City",
    "Stoke":         "Stoke City",
    "Swansea":       "Swansea City",
    "Tottenham":     "Tottenham Hotspur",
    "West Brom":     "West Bromwich Albion",
    "West Ham":      "West Ham United",
    # Bundesliga
    "Darmstadt":     "Darmstadt 98",
    "Dortmund":      "Borussia Dortmund",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln":       "FC Köln",
    "Hamburg":       "Hamburger SV",
    "Hannover":      "Hannover 96",
    "Hertha":        "Hertha Berlin",
    "Leverkusen":    "Bayer Leverkusen",
    "M'gladbach":    "Borussia Mönchengladbach",
    "Mainz":         "FSV Mainz 05",
    "Stuttgart":     "VfB Stuttgart",
    # La Liga
    "Ath Bilbao":    "Athletic Club",
    "Ath Madrid":    "Atlético Madrid",
    "Betis":         "Real Betis",
    "Celta":         "Celta Vigo",
    "Espanol":       "Espanyol",
    "La Coruna":     "RC Deportivo La Coruña",
    "Levante":       "Levante UD",
    "Malaga":        "Málaga",
    "Sociedad":      "Real Sociedad",
    "Sp Gijon":      "Sporting Gijón",
    "Vallecano":     "Rayo Vallecano",
    # Serie A
    "Inter":         "Inter Milan",
    "Milan":         "AC Milan",
    "Roma":          "AS Roma",
    "Verona":        "Hellas Verona",
    # Ligue 1
    "Ajaccio GFCO":  "Gazélec Ajaccio",
    "Monaco":        "AS Monaco",
    "Nice":          "OGC Nice",
    "Paris SG":      "Paris Saint-Germain",
    "Reims":         "Stade de Reims",
    "St Etienne":    "Saint-Étienne",
}


# ── StatsBomb helpers ─────────────────────────────────────────────────────────

def get_available_competitions() -> pd.DataFrame:
    """Return all competitions available in the StatsBomb open-data API."""
    from statsbombpy import sb
    return sb.competitions()


def load_statsbomb_competition(
    competition_id: int,
    season_id: int,
    competition_name: Optional[str] = None,
    match_ids: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all matches and shots for a StatsBomb competition/season.

    Parameters
    ----------
    competition_id : int
        StatsBomb competition ID.
    season_id : int
        StatsBomb season ID.
    competition_name : str, optional
        Label for this competition. Inferred from the API if not provided.
    match_ids : sequence of int, optional
        Restrict loading to these specific match IDs.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    shots_df : pd.DataFrame
        All shot events enriched with match metadata.
    matches_df : pd.DataFrame
        Match-level metadata (home_team, away_team, home_score, away_score, …).
    """
    from statsbombpy import sb
    try:
        from statsbombpy.api_client import NoAuthWarning
        _warn_filter = ("ignore", NoAuthWarning)
    except ImportError:
        _warn_filter = ("ignore",)

    with warnings.catch_warnings():
        warnings.simplefilter(*_warn_filter)
        matches = sb.matches(competition_id=competition_id, season_id=season_id)

    if matches.empty:
        raise ValueError(
            f"No matches found for competition_id={competition_id}, season_id={season_id}."
        )

    if match_ids is not None:
        matches = matches[matches["match_id"].isin(match_ids)].copy()

    # Infer competition name from data if not provided
    if competition_name is None:
        if "competition" in matches.columns:
            competition_name = str(matches["competition"].iloc[0])
        else:
            competition_name = f"competition_{competition_id}"

    # Extract shots for every match
    if verbose:
        try:
            from tqdm.notebook import tqdm as tqdm_nb
            iterator = tqdm_nb(matches.iterrows(), total=len(matches), desc=competition_name)
        except ImportError:
            from tqdm import tqdm
            iterator = tqdm(matches.iterrows(), total=len(matches), desc=competition_name)
    else:
        iterator = matches.iterrows()

    all_shots: list[pd.DataFrame] = []
    with warnings.catch_warnings():
        warnings.simplefilter(*_warn_filter)
        for _, row in iterator:
            shots = _extract_shots_from_match(
                match_id=int(row["match_id"]),
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                match_date=str(row.get("match_date", "")),
                competition_name=competition_name,
                season_name=str(row.get("season", "")),
                competition_stage=str(row["competition_stage"])
                    if pd.notna(row.get("competition_stage")) else None,
                match_week=int(row["match_week"])
                    if pd.notna(row.get("match_week")) else None,
            )
            if not shots.empty:
                all_shots.append(shots)

    shots_df = pd.concat(all_shots, ignore_index=True) if all_shots else pd.DataFrame()
    return shots_df, matches


def _extract_shots_from_match(
    match_id: int,
    home_team: str,
    away_team: str,
    match_date: str,
    competition_name: str,
    season_name: str,
    competition_stage: Optional[str],
    match_week: Optional[int],
) -> pd.DataFrame:
    from statsbombpy import sb

    try:
        events = sb.events(match_id=match_id)
    except Exception as exc:
        warnings.warn(f"Could not fetch events for match {match_id}: {exc}")
        return pd.DataFrame()

    shots = events[events["type"] == "Shot"].copy()
    if shots.empty:
        return shots

    shots["match_id"]          = match_id
    shots["home_team"]         = home_team
    shots["away_team"]         = away_team
    shots["match_date"]        = match_date
    shots["competition_name"]  = competition_name
    shots["season_name"]       = season_name
    shots["competition_stage"] = competition_stage
    shots["match_week"]        = match_week

    keep = [
        "match_id", "home_team", "away_team", "match_date",
        "competition_name", "season_name", "competition_stage", "match_week",
        "id", "index", "period", "timestamp", "minute", "second",
        "team", "player", "position", "location",
        "shot_statsbomb_xg", "shot_outcome", "shot_type", "shot_body_part",
        "shot_technique", "shot_first_time", "shot_one_on_one",
        "shot_end_location", "shot_freeze_frame",
    ]
    return shots[[c for c in keep if c in shots.columns]]


# ── Match-level aggregation ───────────────────────────────────────────────────

def process_shots_to_match_summary(
    shots_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate shot-level data into a match-level summary DataFrame.

    Goal counts are taken from `matches_df` (home_score / away_score) to
    correctly handle own goals that may not appear as regular shot events.

    Returns one row per match with columns defined in the module docstring.
    """
    shots = shots_df.copy()
    shots["is_goal"] = shots["shot_outcome"].apply(
        lambda x: 1 if isinstance(x, str) and "Goal" in x else 0
    )

    # Aggregate per match × team
    agg = (
        shots.groupby(["match_id", "team"])
        .agg(
            xg_total=("shot_statsbomb_xg", "sum"),
            shots=("shot_statsbomb_xg", "count"),
            xg_values=("shot_statsbomb_xg", list),
        )
        .reset_index()
    )

    # Build match-level records
    records: list[dict] = []
    for match_id, grp in agg.groupby("match_id"):
        mrow = matches_df[matches_df["match_id"] == match_id]
        if mrow.empty:
            continue
        mrow = mrow.iloc[0]

        home_team = str(mrow["home_team"])
        away_team = str(mrow["away_team"])

        home = grp[grp["team"] == home_team]
        away = grp[grp["team"] == away_team]

        def _val(sub: pd.DataFrame, col: str, default):
            return sub[col].iloc[0] if not sub.empty else default

        # Authoritative goal counts from match metadata
        home_goals = int(mrow.get("home_score", _val(home, "xg_total", 0)))
        away_goals = int(mrow.get("away_score", _val(away, "xg_total", 0)))
        # Fall back to shot-level counting only if match metadata has no score columns
        if "home_score" not in mrow.index:
            home_goals = int(shots[(shots["match_id"] == match_id) &
                                   (shots["team"] == home_team)]["is_goal"].sum())
            away_goals = int(shots[(shots["match_id"] == match_id) &
                                   (shots["team"] == away_team)]["is_goal"].sum())

        competition_name = _val(home, "xg_total", None)  # placeholder
        competition_name = str(
            mrow.get("competition_name",
                     mrow.get("competition",
                              shots[shots["match_id"] == match_id]["competition_name"].iloc[0]
                              if "competition_name" in shots.columns else ""))
        )
        season_name = str(mrow.get("season", ""))
        comp_stage  = str(mrow.get("competition_stage", "")) or None
        mweek       = int(mrow["match_week"]) if pd.notna(mrow.get("match_week")) else None

        records.append({
            "match_id":          int(match_id),
            "match_date":        str(mrow.get("match_date", "")),
            "competition_name":  competition_name,
            "season_name":       season_name,
            "competition_stage": comp_stage,
            "match_week":        mweek,
            "home_team":         home_team,
            "away_team":         away_team,
            "home_goals":        home_goals,
            "away_goals":        away_goals,
            "home_xg":           float(_val(home, "xg_total", 0.0)),
            "away_xg":           float(_val(away, "xg_total", 0.0)),
            "home_shots":        int(_val(home, "shots", 0)),
            "away_shots":        int(_val(away, "shots", 0)),
            "home_xg_values":    json.dumps(_val(home, "xg_values", [])),
            "away_xg_values":    json.dumps(_val(away, "xg_values", [])),
        })

    return pd.DataFrame(records)


# ── Betting odds helpers ──────────────────────────────────────────────────────

def convert_odds_to_probabilities(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
) -> tuple[float, float, float]:
    """Convert decimal odds to normalised implied probabilities (overround removed)."""
    raw_h = 1.0 / odds_home
    raw_d = 1.0 / odds_draw
    raw_a = 1.0 / odds_away
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def load_football_data_odds(
    csv_path: Path,
    competition_name: str,
    team_name_map: Optional[dict[str, str]] = None,
    odds_cols: tuple[str, str, str] = ("B365H", "B365D", "B365A"),
    date_col: str = "Date",
    date_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and process a football-data.co.uk CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file (e.g. ``data/odds/E0.csv``).
    competition_name : str
        Label matching the StatsBomb competition_name (used for merging).
    team_name_map : dict, optional
        Map from football-data team names to StatsBomb names.
        Defaults to TEAM_NAME_MAP_DEFAULT.
    odds_cols : (str, str, str)
        Column names for (home, draw, away) decimal odds.
    date_col : str
        Column name for match date.
    date_format : str, optional
        strptime format string; if None, pandas infers format (day-first).

    Returns
    -------
    pd.DataFrame with columns:
        match_date, competition_name, home_team, away_team,
        odds_home, odds_draw, odds_away,
        betting_p_home, betting_p_draw, betting_p_away
    """
    if team_name_map is None:
        team_name_map = TEAM_NAME_MAP_DEFAULT

    h_col, d_col, a_col = odds_cols
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[h_col, d_col, a_col]).copy()

    df["HomeTeam"] = df["HomeTeam"].replace(team_name_map)
    df["AwayTeam"]  = df["AwayTeam"].replace(team_name_map)

    probs = df.apply(
        lambda r: pd.Series(
            convert_odds_to_probabilities(r[h_col], r[d_col], r[a_col]),
            index=["betting_p_home", "betting_p_draw", "betting_p_away"],
        ),
        axis=1,
    )
    df = pd.concat([df, probs], axis=1)
    df["competition_name"] = competition_name

    if date_format:
        df["match_date"] = pd.to_datetime(df[date_col], format=date_format).dt.strftime("%Y-%m-%d")
    else:
        df["match_date"] = pd.to_datetime(df[date_col], dayfirst=True).dt.strftime("%Y-%m-%d")

    df = df.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team"})
    return df[
        ["match_date", "competition_name", "home_team", "away_team",
         h_col, d_col, a_col,
         "betting_p_home", "betting_p_draw", "betting_p_away"]
    ].rename(columns={h_col: "odds_home", d_col: "odds_draw", a_col: "odds_away"}).copy()


def merge_odds_with_matches(
    match_summary: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join betting odds into match_summary on (match_date, home_team, away_team).

    Adds betting_p_home, betting_p_draw, betting_p_away columns (NaN when unmatched).
    If these columns already exist they are replaced.
    """
    drop_cols = [c for c in ["betting_p_home", "betting_p_draw", "betting_p_away"]
                 if c in match_summary.columns]
    ms = match_summary.drop(columns=drop_cols)

    odds_slim = (
        odds_df[["match_date", "home_team", "away_team",
                 "betting_p_home", "betting_p_draw", "betting_p_away"]]
        .drop_duplicates(subset=["match_date", "home_team", "away_team"])
    )

    merged = ms.merge(odds_slim, on=["match_date", "home_team", "away_team"], how="left")
    n = merged["betting_p_home"].notna().sum()
    print(f"Betting odds matched for {n}/{len(merged)} matches.")
    return merged


# ── Persistence ───────────────────────────────────────────────────────────────

def save_competition_data(
    match_summary: pd.DataFrame,
    output_dir: Path,
    competition_slug: str,
) -> Path:
    """
    Save a processed match-summary DataFrame to ``output_dir/<slug>.parquet``.

    ``home_xg_values`` / ``away_xg_values`` must already be JSON strings
    (they are when returned by process_shots_to_match_summary).

    Returns the output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_slug(competition_slug)}.parquet"
    match_summary.to_parquet(path, index=False)
    print(f"Saved {len(match_summary)} matches → {path}")
    return path


def load_competition_data(path: Path) -> pd.DataFrame:
    """
    Load a competition parquet file and parse JSON xG-value lists.

    Returns a DataFrame where ``home_xg_values`` and ``away_xg_values`` are
    Python lists of floats (not JSON strings).
    """
    df = pd.read_parquet(path)
    df["home_xg_values"] = df["home_xg_values"].apply(json.loads)
    df["away_xg_values"] = df["away_xg_values"].apply(json.loads)
    return df


# ── Internal ──────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    """Lower-case, replace non-alphanumeric runs with underscores."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
