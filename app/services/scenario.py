"""
Historical stress scenario analysis.

Applies named scenario return shocks (based on actual historical crisis periods)
to the current portfolio weights and reports projected portfolio impact.

Each ticker is mapped to an asset class; the scenario shock for that class
is applied to that ticker's weight.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Historical scenario return shocks by asset class
# (approximate peak-to-trough or period returns for each crisis)
# ---------------------------------------------------------------------------
SCENARIOS: dict[str, dict[str, float]] = {
    "2008 Global Financial Crisis": {
        "US Equity": -0.37,
        "Intl Developed Equity": -0.43,
        "Emerging Markets Equity": -0.53,
        "US Aggregate Bonds": 0.07,
        "Gold": 0.05,
        "REITs": -0.39,
        "Short-Term Bonds": 0.06,
        "Cash / Money Market": 0.02,
    },
    "2020 COVID Crash (Q1)": {
        "US Equity": -0.34,
        "Intl Developed Equity": -0.31,
        "Emerging Markets Equity": -0.27,
        "US Aggregate Bonds": 0.03,
        "Gold": -0.01,
        "REITs": -0.40,
        "Short-Term Bonds": 0.02,
        "Cash / Money Market": 0.01,
    },
    "2022 Rate Shock (Full Year)": {
        "US Equity": -0.18,
        "Intl Developed Equity": -0.16,
        "Emerging Markets Equity": -0.22,
        "US Aggregate Bonds": -0.13,
        "Gold": -0.02,
        "REITs": -0.26,
        "Short-Term Bonds": -0.03,
        "Cash / Money Market": 0.03,
    },
    "2013 Taper Tantrum": {
        "US Equity": 0.32,
        "Intl Developed Equity": 0.22,
        "Emerging Markets Equity": -0.05,
        "US Aggregate Bonds": -0.02,
        "Gold": -0.28,
        "REITs": 0.02,
        "Short-Term Bonds": 0.01,
        "Cash / Money Market": 0.01,
    },
    "Equity Bear (-30% hypothetical)": {
        "US Equity": -0.30,
        "Intl Developed Equity": -0.28,
        "Emerging Markets Equity": -0.35,
        "US Aggregate Bonds": 0.05,
        "Gold": 0.08,
        "REITs": -0.30,
        "Short-Term Bonds": 0.02,
        "Cash / Money Market": 0.01,
    },
    "Rising Rates +200bp (hypothetical)": {
        "US Equity": -0.10,
        "Intl Developed Equity": -0.08,
        "Emerging Markets Equity": -0.15,
        "US Aggregate Bonds": -0.16,
        "Gold": -0.05,
        "REITs": -0.20,
        "Short-Term Bonds": -0.04,
        "Cash / Money Market": 0.04,
    },
    "Inflation Spike +3pp (hypothetical)": {
        "US Equity": -0.08,
        "Intl Developed Equity": -0.06,
        "Emerging Markets Equity": -0.04,
        "US Aggregate Bonds": -0.10,
        "Gold": 0.12,
        "REITs": 0.03,
        "Short-Term Bonds": -0.02,
        "Cash / Money Market": 0.04,
    },
}

# ---------------------------------------------------------------------------
# Ticker -> Asset class mapping  (best-effort; unknown tickers default to US Equity)
# ---------------------------------------------------------------------------
TICKER_TO_ASSET_CLASS: dict[str, str] = {
    # US broad equity
    "SPY": "US Equity", "IVV": "US Equity", "VOO": "US Equity",
    "VTI": "US Equity", "SCHB": "US Equity", "ITOT": "US Equity",
    # US sectors / style (treated as US equity)
    "QQQ": "US Equity", "XLK": "US Equity", "XLF": "US Equity",
    "XLE": "US Equity", "XLV": "US Equity", "XLY": "US Equity",
    "IWM": "US Equity", "VTV": "US Equity", "VUG": "US Equity",
    "MTUM": "US Equity", "QUAL": "US Equity",
    # Intl developed
    "EFA": "Intl Developed Equity", "VEA": "Intl Developed Equity",
    "SCHF": "Intl Developed Equity", "VXUS": "Intl Developed Equity",
    "IEFA": "Intl Developed Equity",
    # Emerging markets
    "EEM": "Emerging Markets Equity", "VWO": "Emerging Markets Equity",
    "IEMG": "Emerging Markets Equity",
    # US bonds (aggregate / intermediate)
    "AGG": "US Aggregate Bonds", "BND": "US Aggregate Bonds",
    "SCHZ": "US Aggregate Bonds", "IUSB": "US Aggregate Bonds",
    # Short-term bonds
    "SHY": "Short-Term Bonds", "SCHO": "Short-Term Bonds",
    "BSV": "Short-Term Bonds", "VGSH": "Short-Term Bonds",
    "BIL": "Cash / Money Market",
    # Gold / commodities
    "GLD": "Gold", "IAU": "Gold", "SGOL": "Gold",
    "GSG": "US Equity",   # commodity ETF – proxy as equity
    # REITs
    "VNQ": "REITs", "IYR": "REITs", "SCHH": "REITs", "REM": "REITs",
    # Individual stocks default to US Equity via fallback
}

_DEFAULT_ASSET_CLASS = "US Equity"


def run_scenarios(portfolio_weights: dict[str, float]) -> list[dict]:
    """
    Apply all historical stress scenarios to *portfolio_weights*.

    Returns a list of scenario results sorted from worst to best projected return.
    Each result includes per-ticker detail (weight, asset class, shock applied,
    contribution to portfolio return).
    """
    results = []
    for name, shocks in SCENARIOS.items():
        port_return = 0.0
        ticker_detail = []
        for ticker, weight in portfolio_weights.items():
            asset_class = TICKER_TO_ASSET_CLASS.get(ticker.upper(), _DEFAULT_ASSET_CLASS)
            shock = shocks.get(asset_class, shocks.get(_DEFAULT_ASSET_CLASS, 0.0))
            contribution = weight * shock
            port_return += contribution
            ticker_detail.append(
                {
                    "ticker": ticker,
                    "weight": round(weight, 4),
                    "asset_class": asset_class,
                    "shock_pct": round(shock * 100, 1),
                    "contribution_pct": round(contribution * 100, 2),
                }
            )
        results.append(
            {
                "scenario": name,
                "projected_return_pct": round(port_return * 100, 2),
                "detail": ticker_detail,
            }
        )

    results.sort(key=lambda r: r["projected_return_pct"])
    return results
