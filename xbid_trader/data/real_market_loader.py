"""Loader for REAL Greek market data, joined with the committed battery plan.

Merges three workbooks by 15-minute timestamp:
  * the battery plan (``Charge_MW/Discharge_MW/Energy_MWh``) — optionally
    co-optimised with the capacity market, in which case it also carries
    ``DAM_Charge_MW/DAM_Discharge_MW`` (the *tradeable* day-ahead position) and
    the awards ``FCR_UP_MW, FCR_DN_MW, aFRR_Up_MW, aFRR_Dn_MW, mFRR_Up_MW,
    mFRR_Dn_MW`` with their clearing prices;
  * ``Energy_Market_Data.xlsx`` (real DAM price + real XBID Weighted-Average /
    MIN / MAX price per quarter);
  * ``Imbalance_Price*.xlsx`` (real imbalance settlement price).

Capacity handling
-----------------
When the plan is co-optimised with reserves, the physical dispatch of a quarter
is::

    physical = tradeable DAM position + reserve activation

Only the tradeable part can be traded on XBID; the activation is dispatched by
the TSO. The activation volume is assumed to be ``activation_rate`` (default
0.40) of the awarded capacity — i.e. 40 % of the committed reserve is called::

    activation[i] = activation_rate * (up_mw[i] - dn_mw[i]) * 0.25   MWh

The awards additionally reserve power head-room and SoC energy, both of which
the controller honours.

``scale_mw`` multiplies every power/energy column, for workbooks normalised to a
1 MW / 2 MWh unit (use 50 for a real 50 MW / 100 MWh battery).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SLOT_HOURS = 0.25

RESERVE_UP_COLS = ["FCR_UP_MW", "aFRR_Up_MW", "mFRR_Up_MW"]
RESERVE_DN_COLS = ["FCR_DN_MW", "aFRR_Dn_MW", "mFRR_Dn_MW"]
RESERVE_PRICE_PAIRS = [
    ("FCR_UP_MW", "FCR_Up_Price"), ("FCR_DN_MW", "FCR_Dn_Price"),
    ("aFRR_Up_MW", "aFRR_Up_Price"), ("aFRR_Dn_MW", "aFRR_Dn_Price"),
    ("mFRR_Up_MW", "mFRR_Up_Price"), ("mFRR_Dn_MW", "mFRR_Dn_Price"),
]


@dataclass
class RealDay:
    dam_price: np.ndarray        # real DAM price €/MWh
    xbid_price: np.ndarray       # real XBID VWAP €/MWh (NaN where no XBID trade)
    xbid_spread: np.ndarray      # real (MAX-MIN)/2 €/MWh (0 where unknown)
    imb_price: np.ndarray        # real imbalance price €/MWh
    dam_net: np.ndarray          # TRADEABLE committed net (MWh, + discharge / − charge)
    tradeable: np.ndarray        # bool: XBID liquidity exists this quarter
    # capacity-market extras (zero-filled when the plan is energy-only)
    up_mw: np.ndarray = field(default=None)                # awarded UP reserve (MW)
    dn_mw: np.ndarray = field(default=None)                # awarded DOWN reserve (MW)
    reserve_activation: np.ndarray = field(default=None)   # MWh/slot, fixed
    reserve_revenue: float = 0.0                # € already earned in capacity markets
    plan_soc: np.ndarray = field(default=None)  # SoC path of the source plan (MWh)
    soc_start: Optional[float] = None           # SoC entering the day (MWh)
    dam_settled: np.ndarray = field(default=None)   # what settles in DAM (MWh/slot)

    def __post_init__(self):
        n = len(self.dam_price)
        if self.dam_settled is None:
            self.dam_settled = self.dam_net
        if self.up_mw is None:
            self.up_mw = np.zeros(n)
        if self.dn_mw is None:
            self.dn_mw = np.zeros(n)
        if self.reserve_activation is None:
            self.reserve_activation = np.zeros(n)

    def physical(self, tradeable_net: Optional[np.ndarray] = None) -> np.ndarray:
        """Physical dispatch = tradeable position + reserve activation."""
        q = self.dam_net if tradeable_net is None else tradeable_net
        return np.asarray(q, dtype=float) + self.reserve_activation

    @property
    def has_capacity(self) -> bool:
        return bool(np.any(self.up_mw) or np.any(self.dn_mw))


def _num(s):
    ser = pd.Series(s)
    if ser.dtype == object:
        ser = ser.where(ser != "-", np.nan)
    return pd.to_numeric(ser, errors="coerce")


def load_real_market_newfmt(schedule_xlsx: str, energy_xlsx: str, imbalance_xlsx: str,
                            months: Optional[List[str]] = None,
                            activation_rate: float = 0.20,
                            eta_charge: float = 0.95, eta_discharge: float = 0.95):
    """Backtest loader for the `BESS_Optimized_Schedule_*.xlsx` (new) format.

    Same idea as ``load_real_market`` but for the new column names, and it merges
    the schedule with the REAL XBID prints in ``Energy_Market_Data.xlsx`` (VWAP +
    MIN/MAX for the spread) and the real imbalance price, by timestamp. Values are
    real (no scaling). Returns {day -> RealDay} for the overlapping days.
    """
    sch = pd.read_excel(schedule_xlsx, sheet_name="BESS_Schedule")
    sch["dt"] = pd.to_datetime(sch["Delivery_Interval"], errors="coerce")
    sch = sch[sch["dt"].notna()].copy()

    em = pd.read_excel(energy_xlsx, sheet_name="Dataframe")
    em["dt"] = pd.to_datetime(em["Start date (Greek Time)"], errors="coerce")
    em = em.rename(columns={"XBID Weighted Average Price [€/MWh]": "xbid",
                            "XBID MIN Price [€/MWh]": "xmin",
                            "XBID MAX Price [€/MWh]": "xmax",
                            "DAM Price [€/MWh]": "dam"})
    for c in ("dam", "xbid", "xmin", "xmax"):
        em[c] = _num(em[c])

    imb = pd.read_excel(imbalance_xlsx)
    price_col = [c for c in imb.columns if c not in ("version", "tradingPeriod")][0]
    imb["dt"] = pd.to_datetime(imb["tradingPeriod"], errors="coerce", format="mixed")
    imb = imb.dropna(subset=["dt"]).rename(columns={price_col: "imb"})
    imb["imb"] = _num(imb["imb"])

    up_col = "Total_Capacity_Up_MW"
    dn_col = "Total_Capacity_Dn_MW"
    keep = ["dt", "DAM_Price_EUR_MWh", "Physical_Charge_MW", "Physical_Discharge_MW",
            "DAM_Charge_MW", "DAM_Discharge_MW", "SOE_Capacity_MWh", up_col, dn_col]
    keep = [c for c in keep if c in sch.columns]
    mg = (sch[keep].merge(em[["dt", "dam", "xbid", "xmin", "xmax"]], on="dt")
                   .merge(imb[["dt", "imb"]], on="dt"))
    mg["day"] = mg["dt"].dt.strftime("%Y-%m-%d")
    mg["month"] = mg["dt"].dt.strftime("%Y-%m")
    if months:
        mg = mg[mg["month"].isin(set(months))]
    mg = mg.sort_values("dt")

    by_day: Dict[str, RealDay] = {}
    for day, g in mg.groupby("day"):
        g = g.sort_values("dt")
        n = len(g)
        dam_mkt = g["dam"].to_numpy(float)
        dam_sched = _num(g["DAM_Price_EUR_MWh"]).to_numpy(float)
        dam_price = np.where(np.isfinite(dam_mkt), dam_mkt, dam_sched)

        phys = (_num(g["Physical_Discharge_MW"]).fillna(0).to_numpy(float)
                - _num(g["Physical_Charge_MW"]).fillna(0).to_numpy(float)) * SLOT_HOURS
        settled = (_num(g["DAM_Discharge_MW"]).fillna(0).to_numpy(float)
                   - _num(g["DAM_Charge_MW"]).fillna(0).to_numpy(float)) * SLOT_HOURS

        up = _num(g[up_col]).fillna(0).to_numpy(float) if up_col in g else np.zeros(n)
        dn = _num(g[dn_col]).fillna(0).to_numpy(float) if dn_col in g else np.zeros(n)
        activation = activation_rate * (up - dn) * SLOT_HOURS

        plan_soc = soc_start = None
        if "SOE_Capacity_MWh" in g.columns:
            plan_soc = _num(g["SOE_Capacity_MWh"]).to_numpy(float)
            q0 = float(phys[0])
            soc_start = float(plan_soc[0]) - (eta_charge * max(-q0, 0.0)
                                              - max(q0, 0.0) / eta_discharge)

        xb = g["xbid"].to_numpy(float)
        # guard against data-error outliers (e.g. a VWAP of 1.3e8): treat any
        # price outside a sane band as "no XBID market" for that quarter.
        xb = np.where((xb >= -500.0) & (xb <= 1000.0), xb, np.nan)
        xmin = g["xmin"].to_numpy(float)
        xmax = g["xmax"].to_numpy(float)
        spread = np.where(np.isfinite(xmax) & np.isfinite(xmin), (xmax - xmin) / 2.0, 0.0)

        by_day[day] = RealDay(
            dam_price=dam_price, xbid_price=xb,
            xbid_spread=np.nan_to_num(spread, nan=0.0),
            imb_price=np.nan_to_num(g["imb"].to_numpy(float)),
            dam_net=phys, tradeable=np.isfinite(xb),
            up_mw=up, dn_mw=dn, reserve_activation=activation,
            reserve_revenue=0.0, plan_soc=plan_soc, soc_start=soc_start,
            dam_settled=settled,
        )
    return by_day


def load_real_market(schedule_xlsx: str, energy_xlsx: str, imbalance_xlsx: str,
                     months: Optional[List[str]] = None,
                     scale_mw: float = 1.0,
                     activation_rate: float = 0.40,
                     eta_charge: float = 0.95,
                     eta_discharge: float = 0.95):
    sch = pd.read_excel(schedule_xlsx, sheet_name="Results")
    dt = pd.to_datetime(sch["Date_Hour"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(sch["Date_Hour"], errors="coerce")
    sch["dt"] = dt
    sch = sch[sch["dt"].notna()].copy()          # drop totals / blank trailing rows

    cols = set(sch.columns)
    has_cap = {"DAM_Charge_MW", "DAM_Discharge_MW"} <= cols
    up_cols = [c for c in RESERVE_UP_COLS if c in cols]
    dn_cols = [c for c in RESERVE_DN_COLS if c in cols]

    em = pd.read_excel(energy_xlsx, sheet_name="Dataframe")
    em["dt"] = pd.to_datetime(em["Start date (Greek Time)"], errors="coerce")
    em = em.rename(columns={
        "DAM Price [€/MWh]": "dam",
        "XBID Weighted Average Price [€/MWh]": "xbid",
        "XBID MIN Price [€/MWh]": "xmin",
        "XBID MAX Price [€/MWh]": "xmax",
    })
    for c in ("dam", "xbid", "xmin", "xmax"):
        em[c] = _num(em[c])

    imb = pd.read_excel(imbalance_xlsx)
    price_col = [c for c in imb.columns if c not in ("version", "tradingPeriod")][0]
    imb["dt"] = pd.to_datetime(imb["tradingPeriod"], errors="coerce", format="mixed")
    imb = imb.dropna(subset=["dt"]).rename(columns={price_col: "imb"})
    imb["imb"] = _num(imb["imb"])

    keep = ["dt", "Price", "Charge_MW", "Discharge_MW"]
    if "Energy_MWh" in cols:
        keep.append("Energy_MWh")
    if has_cap:
        keep += ["DAM_Charge_MW", "DAM_Discharge_MW"]
    keep += up_cols + dn_cols
    keep += [px for _, px in RESERVE_PRICE_PAIRS if px in cols]

    mg = (sch[keep].merge(em[["dt", "dam", "xbid", "xmin", "xmax"]], on="dt")
                   .merge(imb[["dt", "imb"]], on="dt"))
    mg["day"] = mg["dt"].dt.strftime("%Y-%m-%d")
    mg["month"] = mg["dt"].dt.strftime("%Y-%m")
    if months:
        mg = mg[mg["month"].isin(set(months))]
    mg = mg.sort_values("dt")

    k = float(scale_mw)
    by_day: Dict[str, RealDay] = {}
    for day, g in mg.groupby("day"):
        g = g.sort_values("dt")
        n = len(g)

        dam_mkt = g["dam"].to_numpy(float)
        dam_sched = _num(g["Price"]).to_numpy(float)
        dam_price = np.where(np.isfinite(dam_mkt), dam_mkt, dam_sched)

        # The committed OPERATION is Charge/Discharge: it reproduces the plan's
        # Energy_MWh exactly and uses exactly the plan's cycle budget, so it is
        # the physical baseline the intraday layer trades around.
        phys_plan = (_num(g["Discharge_MW"]).fillna(0).to_numpy(float)
                     - _num(g["Charge_MW"]).fillna(0).to_numpy(float)) * SLOT_HOURS * k
        tradeable_net = phys_plan
        # DAM_* is what settles in the day-ahead market -> used for revenue only.
        if has_cap:
            dam_settled = (_num(g["DAM_Discharge_MW"]).fillna(0).to_numpy(float)
                           - _num(g["DAM_Charge_MW"]).fillna(0).to_numpy(float)
                           ) * SLOT_HOURS * k
        else:
            dam_settled = phys_plan

        up = np.zeros(n)
        for c in up_cols:
            up += _num(g[c]).fillna(0).to_numpy(float)
        dn = np.zeros(n)
        for c in dn_cols:
            dn += _num(g[c]).fillna(0).to_numpy(float)
        up *= k
        dn *= k

        # ASSUMPTION: `activation_rate` of the awarded capacity is actually called.
        activation = activation_rate * (up - dn) * SLOT_HOURS

        revenue = 0.0
        for mw_col, px_col in RESERVE_PRICE_PAIRS:
            if mw_col in g.columns and px_col in g.columns:
                revenue += float(np.sum(_num(g[mw_col]).fillna(0).to_numpy(float) * k
                                        * _num(g[px_col]).fillna(0).to_numpy(float)
                                        * SLOT_HOURS))

        plan_soc = None
        soc_start = None
        if "Energy_MWh" in g.columns:
            plan_soc = _num(g["Energy_MWh"]).to_numpy(float) * k
            q0 = float(phys_plan[0])
            soc_start = float(plan_soc[0]) - (eta_charge * max(-q0, 0.0)
                                              - max(q0, 0.0) / eta_discharge)

        xb = g["xbid"].to_numpy(float)
        # guard against data-error outliers (e.g. a VWAP of 1.3e8): treat any
        # price outside a sane band as "no XBID market" for that quarter.
        xb = np.where((xb >= -500.0) & (xb <= 1000.0), xb, np.nan)
        xmin = g["xmin"].to_numpy(float)
        xmax = g["xmax"].to_numpy(float)
        spread = np.where(np.isfinite(xmax) & np.isfinite(xmin), (xmax - xmin) / 2.0, 0.0)

        by_day[day] = RealDay(
            dam_price=dam_price,
            xbid_price=xb,
            xbid_spread=np.nan_to_num(spread, nan=0.0),
            imb_price=np.nan_to_num(g["imb"].to_numpy(float)),
            dam_net=tradeable_net,
            tradeable=np.isfinite(xb),
            up_mw=up, dn_mw=dn,
            reserve_activation=activation,
            reserve_revenue=revenue,
            plan_soc=plan_soc,
            soc_start=soc_start,
            dam_settled=dam_settled,
        )
    return by_day


def coverage_report(by_day) -> str:
    days = sorted(by_day)
    if not days:
        return "no overlapping days"
    tr = np.concatenate([d.tradeable for d in by_day.values()])
    cap = sum(1 for d in by_day.values() if d.has_capacity)
    txt = (f"{len(days)} days ({days[0]}→{days[-1]}), "
           f"XBID-tradeable quarters: {100 * tr.mean():.0f}%")
    if cap:
        txt += f", capacity awards on {cap}/{len(days)} days"
    return txt
