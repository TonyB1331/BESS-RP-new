"""XBID market sources.

One interface, three possible backings:

    SIMULATION  -> ``simulate_xbid_day()``   Monte-Carlo market sampled from the
                                             EMPIRICAL distribution of real Greek
                                             XBID prints. Use this when tomorrow's
                                             DAM + capacities are known but no live
                                             feed exists yet.
    BACKTEST    -> ``historical_xbid_day()`` the real XBID prints from the market
                                             data workbook.
    LIVE        -> (future) build a ``MarketDay`` from the XBID/SIDC gateway; the
                                             controller needs nothing else.

A ``MarketDay`` tells the controller, for every 15-minute product:
  * ``final_price``  the price the product eventually clears at (NaN if it never
    trades intraday -- most products have no XBID liquidity at all),
  * ``half_spread``  half the bid/ask spread paid when crossing the book,
  * ``tradeable``    whether an XBID market exists for that product,
and, via ``observed(t)``, the price *as known at decision step t* -- prices reveal
gradually as each product approaches its own gate, so the controller never sees
the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Optional

import numpy as np

# ==========================================================================
# CALIBRATION — measured on real Greek market data (Oct 2025 - Mar 2026,
# 7,273 quarter-hours that actually printed on XBID).
#
# Rather than inventing separate "spike" and "negative price" mechanisms, the
# simulator samples the EMPIRICAL distribution of the XBID-minus-DAM deviation.
# Spikes, negative prices and quiet quarters therefore occur at exactly their
# real frequency and magnitude.
# ==========================================================================

# Observed (day_mean, day_std) regimes -- bootstrapped, so the joint
# distribution of calm vs violent days is reproduced exactly.
DAY_REGIMES = np.array([
    [0.47, 15.71], [-1.47, 6.02], [6, 14.58], [7.16, 12.78], [17.66, 26.68], [4.65, 39.96],
    [0.78, 23.1], [-9.92, 20.5], [2.69, 18.02], [-14.67, 11.52], [2.3, 12.96], [1.15, 25.35],
    [4.6, 9.26], [-10.89, 19.34], [-4.09, 12.78], [2.25, 10.1], [7.55, 13.63], [-10.32, 11.92],
    [0.07, 20.75], [1.1, 14.04], [-3.51, 10.72], [11.86, 32.72], [-28.2, 19.05], [6.64, 26.83],
    [-3.75, 15.71], [6.93, 13.18], [-3.09, 9.71], [-3.6, 10.41], [4.34, 15.4], [7.48, 11.79],
    [-14.96, 19.17], [-10.76, 9.2], [1.97, 11.86], [3.09, 5.68], [-14.36, 6.9], [-11.45, 13.41],
    [4.28, 11.87], [-2.48, 18.29], [7.15, 14.44], [13.97, 11.34], [-0.86, 9.44], [-2.86, 2.86],
    [13.23, 15.43], [-5.32, 13.55], [11.37, 12.06], [27.14, 30.21], [11.53, 12.11], [-0.13, 15.55],
    [26.2, 38.18], [7.33, 16.36], [-0.38, 15.59], [-19.83, 16.22], [2.06, 8.39], [7.53, 12.86],
    [-11.55, 10.86], [12.82, 22.19], [5.7, 11.4], [11.32, 15.47], [-2.34, 14.21], [-4.06, 8.15],
    [-13.88, 8.92], [-0.5, 5.27], [-6.42, 6.14], [13.73, 14.09], [1.67, 9.64], [14.47, 25.56],
    [-2.28, 2.72], [5.31, 9.26], [10.18, 9.3], [28.85, 23.53], [24.98, 21.16], [-3.43, 4.84],
    [11.12, 16.49], [-10.1, 10.71], [5.16, 8.74], [9.71, 19.3], [6.15, 11.32], [0.53, 9.12],
    [-11.85, 16.68], [0.86, 6.95], [-4.88, 7.2], [3.76, 4.41], [2.79, 10.82], [0.57, 12.71],
    [-4.78, 16.95], [-0.51, 5.91], [-2.05, 10.89], [-27.23, 11.31], [16.69, 9.43], [37.69, 24.1],
    [-4.04, 28.2], [35.96, 30.61], [6, 24.76], [5.2, 32.94], [3.13, 10.96], [2.27, 9.01],
    [-8.21, 19.66], [6.14, 13.95], [6.47, 8.4], [11.03, 17.06], [9.43, 8.85], [4.24, 12.91],
    [5.34, 24.91], [10.01, 19.05], [31.53, 30.96], [77.4, 86.35], [-2.44, 16.23], [2.09, 7.83],
    [-1.68, 17.77], [-22.28, 18.27], [-27.44, 31.09], [-14.37, 22.84], [-10.04, 15.12], [6.56, 15.42],
    [2.47, 21.45], [-11.72, 13.25], [-8.44, 13.54], [-1.78, 16.66], [17.8, 29.28], [6.76, 10.96],
    [1.65, 24.03], [1.01, 19.78], [14.44, 18], [-1.13, 24.82], [7.51, 24.8], [4.14, 6.7],
    [-6.44, 8.99], [-5.13, 4.46], [7.82, 24.96], [10.43, 23.95], [1.62, 16.11], [-5.54, 13.47],
    [13.83, 34.17], [8.01, 14.46], [-0.06, 10.88], [21.45, 31.08], [23.27, 28.14], [3.3, 21.95],
    [3.69, 16.73], [2.2, 7.17], [-23.6, 12.83], [43.09, 58.86], [-15.97, 32.13], [-11.12, 9.67],
    [-13.44, 15], [-10.04, 12.6], [8.32, 18.15], [-10.44, 17.02], [-4, 15.19], [5.7, 16.09],
    [-0.45, 8.77], [6.64, 14.81], [-13.15, 19.22], [-2.85, 4.38], [-1.52, 12.95], [8.8, 14.75],
    [-8.04, 26.07], [-2.4, 12.99], [26.89, 41.03], [5.22, 13.31], [12.82, 22.36], [-8.19, 9.28],
    [5.72, 15.25], [6.01, 13.71], [0.41, 6.35], [11.85, 22.22], [12.15, 13.63], [1.15, 6.11],
    [-10.09, 20.28], [4.4, 22.63], [-9.59, 11.57], [-2.31, 12.56], [-9.86, 11.92], [-14.48, 16.5],
    [-20.68, 9.26], [-26.99, 18.27], [-8.4, 12.51], [-4.65, 7.32], [-3.95, 11.76], [-1.18, 19.39],
    [25.51, 31.97], [3.32, 10.77], [8.79, 13.41], [0.13, 12.5], [0.48, 9.46], [0.36, 9.27],
    [-2.14, 2.58], [-6.59, 4.22], [-4.63, 4.85], [-6.84, 11.85], [12.28, 17.86], [4.84, 19.81],
    [-26.64, 19.04], [-2.07, 8.98], [-8.5, 13.71], [4.64, 14.48], [-4.14, 9.24], [-2.72, 7.73],
    [-4.52, 7.49], [27.13, 41.6], [14.11, 20.32], [1.38, 13.79], [6.19, 15.75], [-4.37, 14.95],
    [5, 17.33], [19.37, 22.69], [14.35, 18.75], [9.82, 12.91], [29.64, 38.82], [14.19, 25.29],
    [6.51, 20.32], [5.33, 23.02], [7.8, 18.86], [2.6, 8.9], [-6.12, 14.47], [2.18, 8.13],
    [8.04, 12.4], [3.66, 29.92], [5.94, 11.87], [-0.12, 22.2], [-4.07, 19.12], [2.62, 13.12],
    [4.5, 17.03], [-4.31, 15.59], [3.79, 22.53], [20.51, 35.23], [2.45, 13.94], [-14.48, 13.39],
    [3.81, 7.58], [2.22, 7.4], [-4.44, 7.5], [0.61, 12.28], [2.56, 13.03], [2.05, 5.55],
    [-5.38, 7.65], [-2.33, 6.95], [-2.74, 11.66], [-3.9, 9.26], [3.27, 9.38], [0.69, 8.44],
    [2.79, 10.04], [-1.32, 5.95], [10.78, 17.02], [11.97, 21.45], [-0.84, 27.87], [7.62, 30.3],
    [1.47, 5.54], [-0.81, 2.18], [27.52, 37.27], [-8.22, 24.63], [-31.43, 30.79], [0.03, 21.08],
    [-2.38, 4.24], [-4.76, 14.29], [-4.41, 13.07], [-3.79, 12.71], [6.07, 17.75], [2.15, 11.49],
    [-1.47, 3.36], [0.58, 8.58], [-2.31, 3.23], [-2.23, 4.51], [-2.58, 17.68], [3.28, 13.71],
    [-0.23, 10.86], [0.04, 9.44], [0.53, 10.64], [6.23, 12.35], [-7.1, 9.99], [-2.13, 9.28],
    [5.82, 9.75], [-0.51, 8.29], [-3.62, 11.59], [-7.8, 8.99], [-9.67, 8.25],
], dtype=float)

# ---- Within-day shape: standardized deviation (dev - day_mean) / day_std --
SHAPE_Q = np.round(np.arange(0.0, 100.5, 2.0), 1)
SHAPE_V = np.array([
    -5.846, -1.897, -1.528, -1.273, -1.123, -1.007, -0.922, -0.864, -0.801,
    -0.735, -0.683, -0.639, -0.599, -0.558, -0.518, -0.483, -0.446, -0.405,
    -0.363, -0.321, -0.290, -0.256, -0.221, -0.188, -0.149, -0.104, -0.063,
    -0.021, 0.021, 0.060, 0.099, 0.150, 0.196, 0.239, 0.295, 0.340,
    0.401, 0.460, 0.522, 0.593, 0.662, 0.746, 0.849, 0.949, 1.064,
    1.186, 1.338, 1.563, 1.885, 2.425, 6.930,
], dtype=float)

# ---- Liquidity is NOT uniform across the day (night is thin) -------------
LIQ_BY_HOUR = np.array([0.265, 0.237, 0.268, 0.273, 0.283, 0.288, 0.298, 0.291, 0.296, 0.302, 0.496, 0.528, 0.506, 0.523, 0.537, 0.548, 0.560, 0.598, 0.610, 0.635, 0.625, 0.592, 0.599, 0.566],
                       dtype=float)

# ---- Intra-product spread (XBID MAX - MIN, EUR/MWh) ----------------------
SPREAD_PCTL_Q = np.round(np.arange(0.0, 100.5, 5.0), 1)
SPREAD_PCTL_V = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.49, 1.25, 2.50, 4.07, 6.00, 8.69, 11.60, 16.00, 21.17, 33.39, 200.00], dtype=float)

REAL_STATS = dict(
    ar1_rho=0.454,         # lag-1 autocorrelation of the deviation
    dev_scale=1.0,        # scale deviations (stress knob; 1.0 = as observed)
    spread_scale=1.0,     # scale spreads    (stress knob)
    liquidity_scale=1.0,  # scale liquidity  (stress knob)
    price_floor=-500.0,   # market price limits
    price_cap=4000.0,
)
# For reference, the observed statistics this reproduces:
#   recalibrated on 275 days (Oct 2025-Jul 2026): dev std 23.8 EUR/MWh ;
#   |dev|>10 in 41.9% of quarters ; liquidity 44.7% ; spread median 0.5 ;
#   negative XBID price in 4.3% ; median spread 1.0 EUR/MWh.


def _empirical_from_uniform(u: np.ndarray, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Inverse-CDF sampling: map U(0,1) draws onto an empirical distribution."""
    return np.interp(np.clip(u, 0.0, 1.0) * 100.0, q, v)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


@dataclass
class MarketDay:
    """The XBID market for one delivery day, as the controller sees it."""
    dam_price: np.ndarray            # anchor: the product's ex-ante value
    final_price: np.ndarray          # eventual XBID clearing price (NaN if illiquid)
    half_spread: np.ndarray          # EUR/MWh paid when crossing the book
    tradeable: np.ndarray            # bool: an XBID market exists
    reveal_window: int = 48          # quarters before its gate a product reveals over
    source: str = "simulation"

    def observed(self, t: int) -> np.ndarray:
        """Prices as known at decision step ``t`` (gradual revelation).

        A product sits at its DAM value until ``reveal_window`` quarters before
        its own gate, then converges to its eventual XBID price.

        >>> LIVE INTEGRATION POINT <<<  replace with the live order-book snapshot.
        """
        idx = np.arange(len(self.dam_price))
        prog = np.clip((t - (idx - self.reveal_window)) / self.reveal_window, 0.0, 1.0)
        dev = np.where(self.tradeable,
                       np.nan_to_num(self.final_price) - self.dam_price, 0.0)
        return self.dam_price + prog * dev

    def liquidity_pct(self) -> float:
        return 100.0 * float(np.mean(self.tradeable))


# ==========================================================================
# SIMULATION — sample an XBID market from the empirical distribution
# ==========================================================================
def simulate_xbid_day(dam_price: np.ndarray, rng: np.random.Generator,
                      stats: Optional[dict] = None,
                      reveal_window: int = 48) -> MarketDay:
    """Draw one plausible XBID market for a day, given its DAM price curve.

    The next-day workflow: the DAM position and capacity awards are committed,
    the XBID market is not. The market is sampled hierarchically, the way the
    real one behaves:

      1. a DAY REGIME is drawn -- some days are calm, some violent (the per-day
         deviation mean and std both come from their empirical distributions);
      2. WITHIN the day, deviations follow an AR(1)-correlated standardized
         shape, mapped through the empirical (fat-tailed) shape distribution;
      3. LIQUIDITY follows the real hour-of-day profile -- nights are thin,
         afternoons trade;
      4. SPREADS are drawn from their empirical distribution.

    Negative prices and spikes are not special-cased: they fall out of
    ``price = DAM + deviation`` at exactly their real frequency.
    """
    s = {**REAL_STATS, **(stats or {})}
    dam = np.asarray(dam_price, dtype=float)
    n = len(dam)

    # 1. the day's regime -- bootstrap an observed (mean, std) pair
    day_mean, day_sig = DAY_REGIMES[rng.integers(len(DAY_REGIMES))]

    # 2. within-day shape: AR(1) latent -> uniforms -> empirical standardized shape
    rho = float(s["ar1_rho"])
    z = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + np.sqrt(1.0 - rho * rho) * e[i]
    shape = _empirical_from_uniform(_normal_cdf(z), SHAPE_Q, SHAPE_V)

    dev = (day_mean + day_sig * shape) * float(s["dev_scale"])
    final = np.clip(dam + dev, s["price_floor"], s["price_cap"])

    # 3. liquidity: real hour-of-day profile (a quarter's hour = i // 4)
    hours = (np.arange(n) // 4) % 24
    p_liq = np.clip(LIQ_BY_HOUR[hours] * float(s["liquidity_scale"]), 0.0, 1.0)
    tradeable = rng.random(n) < p_liq
    final = np.where(tradeable, final, np.nan)

    # 4. spread
    spread = _empirical_from_uniform(rng.random(n), SPREAD_PCTL_Q, SPREAD_PCTL_V)
    half_spread = np.where(tradeable, spread * float(s["spread_scale"]) / 2.0, 0.0)

    return MarketDay(dam_price=dam, final_price=final, half_spread=half_spread,
                     tradeable=tradeable, reveal_window=reveal_window,
                     source="simulation")


# ==========================================================================
# BACKTEST — the real XBID prints
# ==========================================================================
def historical_xbid_day(dam_price: np.ndarray, xbid_price: np.ndarray,
                        xbid_min: np.ndarray, xbid_max: np.ndarray,
                        reveal_window: int = 48) -> MarketDay:
    """Build a MarketDay from the real XBID prints of a past delivery day."""
    dam = np.asarray(dam_price, dtype=float)
    xb = np.asarray(xbid_price, dtype=float)
    tradeable = np.isfinite(xb)
    lo = np.asarray(xbid_min, dtype=float)
    hi = np.asarray(xbid_max, dtype=float)
    spread = np.where(np.isfinite(hi) & np.isfinite(lo), hi - lo, 0.0)
    half_spread = np.where(tradeable, np.nan_to_num(spread) / 2.0, 0.0)
    return MarketDay(dam_price=dam, final_price=xb, half_spread=half_spread,
                     tradeable=tradeable, reveal_window=reveal_window,
                     source="backtest")
