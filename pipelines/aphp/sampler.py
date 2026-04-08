"""Sampling and random-generation primitives for the AP-HP pipeline.

Pure helper functions ported from ``recode-scenario/utils_v2.py`` (the
``random_date``, ``get_dates_of_stay``, ``get_age``, ``get_names`` and
``sample_from_df`` family). Polars replaces pandas everywhere; pure-Python
randomness uses :mod:`random` and :mod:`numpy.random` so callers can pass
their own seed for reproducibility.
"""

from __future__ import annotations

import datetime as dt
import random as _random
import re
from typing import Any

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def random_date(year: int, exclude_weekends: bool = False, rng: _random.Random | None = None) -> dt.date:
    """Return a uniformly random date in the given calendar year.

    Mirrors ``utils_v2.random_date``: if *exclude_weekends*, keep drawing
    until a weekday lands. Leap years are handled correctly.
    """
    rnd = rng or _random
    days_in_month = {1: 31, 2: 29 if _is_leap(year) else 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    while True:
        month = rnd.randint(1, 12)
        day = rnd.randint(1, days_in_month[month])
        candidate = dt.date(year, month, day)
        if not exclude_weekends or candidate.weekday() < 5:
            return candidate


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def random_date_between(start: dt.date, end: dt.date, rng: _random.Random | None = None) -> dt.date:
    """Return a uniformly random date in ``[start, end]`` (inclusive)."""
    rnd = rng or _random
    delta = (end - start).days
    return start + dt.timedelta(days=rnd.randrange(delta + 1))


def get_dates_of_stay(
    *,
    admission_type: str | None,
    admission_mode: str | None,
    los_mean: float | None = None,
    los_sd: float | None = None,
    los: int | None = None,
    year: int,
    rng: _random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> tuple[dt.date, dt.date]:
    """Generate ``(date_entry, date_discharge)`` for one stay.

    Outpatient stays collapse to a single same-day visit. Inpatient stays draw
    a length of stay from a normal distribution centered on *los_mean* with
    *los_sd* standard deviation (clamped to ``|·|``), then place the entry on
    a random date — weekdays only unless the patient came through emergency.

    Mirrors ``utils_v2.get_dates_of_stay`` (utils_v2.py:61).
    """
    if admission_type == "Outpatient":
        d = random_date(year, exclude_weekends=False, rng=rng)
        return d, d

    if los is None:
        nrng = np_rng or np.random.default_rng()
        mean = los_mean if los_mean is not None else 1
        sd = los_sd if los_sd is not None else 1
        los = int(abs(nrng.normal(mean, sd)))

    exclude_we = admission_mode != "URGENCES"
    date_entry = random_date(year, exclude_weekends=exclude_we, rng=rng)
    return date_entry, date_entry + dt.timedelta(days=los)


# ---------------------------------------------------------------------------
# Age
# ---------------------------------------------------------------------------

_CAGE_RANGE_RE = re.compile(r"\[(\d+)-(\d+)\[")
_CAGE_OPEN_RE = re.compile(r"\[(\d+)-\[")


def extract_age_range(cage: str) -> tuple[int, int] | None:
    """Parse an age class label like ``"[30-40["`` or ``"[80-["`` into a range.

    Returns ``None`` if the string does not match either format. Open ranges
    (``"[80-["``) are capped at 90, matching ``utils_v2.extract_integers_from_cage``.
    """
    if not cage:
        return None
    s = cage.strip()
    m = _CAGE_RANGE_RE.match(s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _CAGE_OPEN_RE.match(s)
    if m:
        return int(m.group(1)), 90
    return None


def random_age(cage: str, rng: _random.Random | None = None) -> int | None:
    """Return a uniformly random integer age within the *cage* class label."""
    rng = rng or _random
    bounds = extract_age_range(cage)
    if bounds is None:
        return None
    return rng.randint(*bounds)


# ---------------------------------------------------------------------------
# Names and demographics
# ---------------------------------------------------------------------------


def interpret_sexe(sexe: int) -> str:
    """``1`` → ``"Masculin"``, anything else → ``"Féminin"`` (legacy mapping)."""
    return "Masculin" if sexe == 1 else "Féminin"


def _titlecase(value: str) -> str:
    if not value:
        return value
    return value[0].upper() + value[1:].lower()


def pick_name(names: pl.DataFrame, gender: int, rng: _random.Random | None = None) -> tuple[str, str]:
    """Sample one ``(first_name, last_name)`` pair for the given gender.

    Mirrors ``generate_scenario.get_names`` (utils_v2.py:457): only picks
    names longer than 3 characters and applies title-casing. *names* must be
    a collected ``pl.DataFrame`` with columns ``prenom``, ``nom``, ``sexe``.
    """
    rng = rng or _random
    seed = rng.randint(0, 2**31 - 1)

    first_pool = names.filter(
        (pl.col("sexe") == gender) & (pl.col("prenom").str.len_chars() > 3)
    )
    last_pool = names.filter(pl.col("nom").str.len_chars() > 3)

    first = first_pool.sample(1, seed=seed).item(0, "prenom")
    last = last_pool.sample(1, seed=seed + 1).item(0, "nom")
    return _titlecase(first), _titlecase(last)


def pick_hospital(hospitals: pl.DataFrame, rng: _random.Random | None = None) -> str:
    """Sample one hospital name from the CHU referential."""
    rng = rng or _random
    return hospitals.sample(1, seed=rng.randint(0, 2**31 - 1)).item(0, "hospital")


# ---------------------------------------------------------------------------
# Profile-conditioned sampling
# ---------------------------------------------------------------------------


def _profile_filter(profile: dict[str, Any], columns: list[str]) -> pl.Expr | None:
    """Build a polars predicate matching the profile values on shared columns."""
    conds = [
        pl.col(k) == v
        for k, v in profile.items()
        if k in columns and v is not None
    ]
    if not conds:
        return None
    expr = conds[0]
    for c in conds[1:]:
        expr = expr & c
    return expr


def sample_conditional(
    df: pl.DataFrame,
    profile: dict[str, Any],
    *,
    nb: int | None = None,
    max_nb: int = 2,
    weight_col: str = "nb",
    distinct_chapter: bool = False,
    chapter_col: str = "icd_secondary_code",
    rng: _random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Profile-conditioned weighted sampling — polars port of ``sample_from_df``.

    Filters *df* down to rows whose values agree with *profile* on every
    shared column, then draws ``nb`` samples (random in ``[0, max_nb]`` if
    *nb* is None) weighted by *weight_col*. When *distinct_chapter* is true,
    successive draws are constrained to come from different ICD chapters
    (first character of *chapter_col*), matching the loop in
    ``utils_v2.sample_from_df`` (utils_v2.py:508–517).

    Returns the sampled rows (without the description-enrichment that the
    original did inline — that lives in :mod:`pipelines.aphp.scenario`).
    """
    rng = rng or _random
    np_rng = np_rng or np.random.default_rng()

    expr = _profile_filter(profile, df.columns)
    df_sel = df.filter(expr) if expr is not None else df

    if df_sel.is_empty():
        return df_sel

    # Default weights to 1 when the column is missing or sums to zero.
    if weight_col not in df_sel.columns or df_sel[weight_col].sum() == 0:
        df_sel = df_sel.with_columns(pl.lit(1).alias(weight_col))

    if nb is not None:
        n_final = min(df_sel.height, nb)
    else:
        n_final = int(np_rng.integers(0, min(df_sel.height, max_nb) + 1))

    if n_final == 0:
        return df_sel.head(0)

    if distinct_chapter and chapter_col in df_sel.columns:
        return _sample_distinct_chapters(
            df_sel, n_final, weight_col, chapter_col, rng
        )

    return df_sel.sample(
        n=n_final, with_replacement=False, seed=rng.randint(0, 2**31 - 1)
    )


def _sample_distinct_chapters(
    df_sel: pl.DataFrame,
    n_final: int,
    weight_col: str,
    chapter_col: str,
    rng: _random.Random,
) -> pl.DataFrame:
    """Iteratively draw rows whose ICD chapter (first char) is unseen so far."""
    pool = df_sel
    chapters: set[str] = set()
    picks: list[pl.DataFrame] = []

    for _ in range(n_final):
        if pool.is_empty():
            break
        pick = pool.sample(
            n=1,
            with_replacement=False,
            seed=rng.randint(0, 2**31 - 1),
        )
        picks.append(pick)
        chapters.add(pick.item(0, chapter_col)[:1])
        pool = pool.filter(~pl.col(chapter_col).str.slice(0, 1).is_in(chapters))

    if not picks:
        return df_sel.head(0)
    return pl.concat(picks)
