from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Segment = Literal["Gold", "Addict", "Organic", "Sinking"]


def assign_segments(
    df: pd.DataFrame,
    *,
    ite_threshold: float = 0.0,
    pae_threshold_mode: Literal["quantile", "fixed"] = "quantile",
    pae_threshold_value: float = 0.7,
) -> pd.DataFrame:
    """
    Quadrant assignment:
      - Gold:    high ITE,     low PAE
      - Addict:  high ITE,     high PAE
      - Organic: low  ITE,    low PAE
      - Sinking: low  ITE,    high PAE
    """
    out = df.copy()

    pae_thr = float(pae_threshold_value)
    if pae_threshold_mode == "quantile":
        pae_thr = float(out["pae"].quantile(pae_thr))

    out["pae_thr"] = pae_thr
    out["ite_high"] = out["ite"] >= float(ite_threshold)
    out["pae_high"] = out["pae"] >= pae_thr

    def to_segment(row) -> Segment:
        ite_high = bool(row["ite_high"])
        pae_high = bool(row["pae_high"])
        if ite_high and (not pae_high):
            return "Gold"
        if ite_high and pae_high:
            return "Addict"
        if (not ite_high) and (not pae_high):
            return "Organic"
        return "Sinking"

    out["segment"] = out.apply(to_segment, axis=1)
    return out.drop(columns=["ite_high", "pae_high", "pae_thr"])


@dataclass(frozen=True)
class OptimizationResult:
    treated_df: pd.DataFrame
    baseline_cost: float
    budget: float
    treated_cost: float
    incremental_gmv: float
    roi: float
    segment_cost_share: dict[str, float]
    segment_increment_share: dict[str, float]


def _fractional_knapsack(
    candidates: pd.DataFrame,
    *,
    budget: float,
    allow_negative_ite: bool = False,
) -> pd.DataFrame:
    """
    Fractional knapsack (greedy by ite/cost).
    Each row is treated with fraction f in [0,1].
    """
    if budget <= 0:
        out = candidates.copy()
        out["treated_frac"] = 0.0
        return out

    c = candidates.copy()
    c = c[c["cost"] > 0]
    if not allow_negative_ite:
        c = c[c["ite"] > 0]
    if c.empty:
        c["treated_frac"] = 0.0
        return c

    c["ratio"] = c["ite"] / c["cost"]
    c = c.sort_values("ratio", ascending=False, kind="mergesort")

    remaining = budget
    treated_frac = np.zeros(len(c), dtype=float)
    costs = c["cost"].to_numpy()

    for i in range(len(c)):
        if remaining <= 1e-12:
            break
        if costs[i] <= remaining + 1e-12:
            treated_frac[i] = 1.0
            remaining -= costs[i]
        else:
            treated_frac[i] = remaining / costs[i]
            remaining = 0.0
            break

    c["treated_frac"] = treated_frac
    return c


def optimize_budget(
    df_with_segments: pd.DataFrame,
    *,
    budget_reduction_pct: float = 10.0,
    addict_accept_cost_share: float = 0.35,
    ite_blocking_mode: Literal["block_negative", "allow_negative"] = "block_negative",
) -> OptimizationResult:
    """
    Budget:
      budget = baseline_cost * (1 - budget_reduction_pct/100)
    """
    baseline_cost = float(df_with_segments["cost"].sum())
    budget = baseline_cost * (1.0 - budget_reduction_pct / 100.0)
    budget = max(0.0, float(budget))

    ite_allow_negative = ite_blocking_mode == "allow_negative"

    # Prepare treatment universe (exclude Sinking by default)
    gold = df_with_segments[df_with_segments["segment"] == "Gold"].copy()
    addict = df_with_segments[df_with_segments["segment"] == "Addict"].copy()
    organic = df_with_segments[df_with_segments["segment"] == "Organic"].copy()
    sinking = df_with_segments[df_with_segments["segment"] == "Sinking"].copy()

    # Threshold Receding (门槛退坡) for Addict:
    # treat only a top share by "cost contribution" after sorting by ite/cost ratio.
    if not addict.empty and addict_accept_cost_share < 1.0:
        if ite_allow_negative:
            addict_ratio = addict["ite"] / addict["cost"].replace(0, np.nan)
        else:
            # For non-negative mode, filter out negative ite before ratio sorting
            addict_pos = addict[addict["ite"] > 0].copy()
            addict_ratio = addict_pos["ite"] / addict_pos["cost"]
            addict_pos = addict_pos.assign(ratio=addict_ratio)
            addict_pos = addict_pos.sort_values("ratio", ascending=False, kind="mergesort")
            addict_ratio_cost = addict_pos["cost"].cumsum() / max(1e-12, float(addict_pos["cost"].sum()))
            keep_mask = addict_ratio_cost <= addict_accept_cost_share
            # ensure at least 1 row if possible
            if keep_mask.any():
                addict = addict_pos.loc[keep_mask].copy()
            else:
                addict = addict_pos.head(1).copy()
        else:
            # allow negative-ite only affects knapsack, not segment filtering
            addict = addict.assign(ratio=addict["ite"] / addict["cost"].replace(0, np.nan)).sort_values(
                "ratio", ascending=False, kind="mergesort"
            )
            cum_share = addict["cost"].cumsum() / max(1e-12, float(addict["cost"].sum()))
            keep_mask = cum_share <= addict_accept_cost_share
            addict = addict.loc[keep_mask].copy() if keep_mask.any() else addict.head(1).copy()

    # Allocate budget in order: Gold -> Addict
    gold_budget = budget
    gold_taken = _fractional_knapsack(gold, budget=gold_budget, allow_negative_ite=ite_allow_negative)
    treated_ids = set(gold_taken["user_id"].tolist()) if not gold_taken.empty else set()
    gold_cost = float((gold_taken["cost"] * gold_taken["treated_frac"]).sum()) if not gold_taken.empty else 0.0
    remaining = max(0.0, budget - gold_cost)

    addict_taken = _fractional_knapsack(addict, budget=remaining, allow_negative_ite=ite_allow_negative)

    # Combine treated fractions back to full df
    treated_df = df_with_segments.copy()
    treated_df["treated_frac"] = 0.0

    if not gold_taken.empty:
        treated_df.loc[treated_df["user_id"].isin(gold_taken["user_id"]), "treated_frac"] = gold_taken["treated_frac"].values
    if not addict_taken.empty:
        treated_df.loc[treated_df["user_id"].isin(addict_taken["user_id"]), "treated_frac"] = addict_taken["treated_frac"].values

    treated_cost = float((treated_df["cost"] * treated_df["treated_frac"]).sum())
    incremental_gmv = float((treated_df["ite"] * treated_df["treated_frac"]).sum())
    roi = incremental_gmv / treated_cost if treated_cost > 0 else 0.0

    seg_cost = treated_df.groupby("segment").apply(lambda x: float((x["cost"] * x["treated_frac"]).sum()))
    seg_inc = treated_df.groupby("segment").apply(lambda x: float((x["ite"] * x["treated_frac"]).sum()))

    # Normalize to "treated" totals (avoid division by 0)
    total_treated_cost = max(1e-12, float(treated_cost))
    total_inc = max(1e-12, float(incremental_gmv))
    segment_cost_share = {k: float(v / total_treated_cost) for k, v in seg_cost.to_dict().items()}
    segment_increment_share = {k: float(v / total_inc) for k, v in seg_inc.to_dict().items()}

    return OptimizationResult(
        treated_df=treated_df,
        baseline_cost=baseline_cost,
        budget=budget,
        treated_cost=treated_cost,
        incremental_gmv=incremental_gmv,
        roi=roi,
        segment_cost_share=segment_cost_share,
        segment_increment_share=segment_increment_share,
    )

