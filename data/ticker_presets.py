"""Standardised DJIA ticker presets for reproducible experiments.

Each preset captures the actual DJIA composition at a specific point in time,
avoiding survivorship bias when back-testing with historical constituents.

Usage in config.yaml
--------------------
Instead of listing tickers explicitly you can write::

    ticker_preset: "djia_2019"

The runner calls ``resolve_tickers(config)`` which replaces the ``tickers``
key with the concrete list.  If ``tickers`` is already a list **and**
``ticker_preset`` is absent the list is used as-is (backward compatible).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

# DJIA composition on 2019-01-01 (test_start in Liu et al. 2020).
# https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average
#
# On 2019-04-02 DowDupont (DWDP) was removed and Dow Inc. (DOW) was added.
# DWDP has been de-listed and DOW only has price history from 2019-03-20 (IPO),
# so neither can cover the full training window (2009-2015).  We therefore omit
# the DowDuPont / Dow Inc. slot, leaving 29 usable tickers.
#
# WBA (Walgreens Boots Alliance) was delisted from major exchanges in 2025 and
# Yahoo Finance no longer serves its historical data.  It is kept in the preset
# for completeness; the download pipeline's ``allow_missing_tickers`` flag will
# gracefully exclude it at runtime when data is unavailable, reducing the set to
# 28 tickers.
#
# On 2020-08-31 Exxon Mobil (XOM), Pfizer (PFE) and Raytheon (RTX, formerly UTX)
# were replaced by Salesforce (CRM), Amgen (AMGN) and Honeywell (HON).
# This preset uses the **pre-reshuffle** composition to avoid survivorship bias
# during the 2019-01 to 2020-09 test window.
DJIA_2019: list[str] = [
    "AAPL", "AXP", "BA", "CAT", "CSCO",
    "CVX", "DIS", "GS", "HD",
    "IBM", "INTC", "JNJ", "JPM", "KO",
    "MCD", "MMM", "MRK", "MSFT", "NKE",
    "PFE", "PG", "RTX", "TRV", "UNH",
    "V", "WBA", "WMT", "XOM", "BAC",
]

# Post-reshuffle DJIA composition (2020-08-31 onwards).
# This is the list that was (incorrectly) used in many reproductions.
# CRM, AMGN, HON replaced XOM, PFE, RTX; DOW and WBA are missing.
# Keeping it as a named preset so users can reproduce the paper's exact
# ticker set while being explicit about the bias.
DJIA_POST_2020_RESHUFFLE: list[str] = [
    "AAPL", "AMGN", "AXP", "BA", "CAT",
    "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "INTC", "JNJ",
    "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "PG", "TRV", "UNH",
    "V", "WMT", "BAC",
]

# Alias used by Liu et al. reproduction — same as post-reshuffle 28-ticker set.
DJIA_LIU_ET_AL_2020: list[str] = list(DJIA_POST_2020_RESHUFFLE)

PRESETS: dict[str, list[str]] = {
    "djia_2019": DJIA_2019,
    "djia_post_2020_reshuffle": DJIA_POST_2020_RESHUFFLE,
    "djia_liu_et_al_2020": DJIA_LIU_ET_AL_2020,
}


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def resolve_tickers(config: dict[str, Any]) -> list[str]:
    """Return the concrete ticker list for *config*, resolving presets.

    Resolution order:

    1. If ``ticker_preset`` is set, look it up in ``PRESETS`` and return it.
       ``tickers`` in the YAML is ignored in this case.
    2. Otherwise fall back to ``config["tickers"]``.
    3. Raise ``ValueError`` if neither is usable.
    """
    preset_name = str(config.get("ticker_preset", "")).strip().lower()
    if preset_name:
        if preset_name not in PRESETS:
            raise ValueError(
                f"Unknown ticker_preset '{preset_name}'. "
                f"Available presets: {sorted(PRESETS.keys())}"
            )
        return list(PRESETS[preset_name])

    tickers = config.get("tickers")
    if isinstance(tickers, list) and len(tickers) > 0:
        return [str(t) for t in tickers]

    raise ValueError(
        "Config must specify either 'ticker_preset' or a non-empty 'tickers' list."
    )


def resolve_benchmark_tickers(config: dict[str, Any]) -> list[str]:
    """Return the ticker list used for the buy-and-hold benchmark.

    Resolution order:

    1. ``benchmark_ticker_preset`` → look up in ``PRESETS``.
    2. ``benchmark_tickers`` → use explicit list.
    3. Fall back to ``resolve_tickers(config)`` (same tickers as the agents).
    """
    preset_name = str(config.get("benchmark_ticker_preset", "")).strip().lower()
    if preset_name:
        if preset_name not in PRESETS:
            raise ValueError(
                f"Unknown ticker_preset '{preset_name}'. "
                f"Available presets: {sorted(PRESETS.keys())}"
            )
        return list(PRESETS[preset_name])

    bm_tickers = config.get("benchmark_tickers")
    if isinstance(bm_tickers, list) and len(bm_tickers) > 0:
        return [str(t) for t in bm_tickers]

    return resolve_tickers(config)
