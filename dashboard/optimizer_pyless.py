from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from dashboard.data import MetricsTable


Segment = Literal["Gold", "Addict", "Organic", "Sinking"]


def _quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def assign_segments(
    metrics: MetricsTable,
    *,
    ite_threshold: float = 0.0,
    pae_threshold_mode: Literal["quantile", "fixed"] = "quantile",
    pae_threshold_value: float = 0.7,
) -> tuple[list[Segment], float]:
    """
    Quadrant assignment:
      - Gold:    high ITE,     low PAE
      - Addict:  high ITE,     high PAE
      - Organic: low  ITE,     low PAE
      - Sinking: low  ITE,     high PAE
    """
    if pae_threshold_mode == "quantile":
        pae_thr = _quantile(metrics.pae, pae_threshold_value)
    else:
        pae_thr = float(pae_threshold_value)

    out: list[Segment] = []
    for ite, pae in zip(metrics.ite, metrics.pae):
        ite_high = ite >= float(ite_threshold)
        pae_high = pae >= pae_thr

        if ite_high and (not pae_high):
            out.append("Gold")
        elif ite_high and pae_high:
            out.append("Addict")
        elif (not ite_high) and (not pae_high):
            out.append("Organic")
        else:
            out.append("Sinking")

    return out, pae_thr


@dataclass(frozen=True)
class OptimizationResult:
    metrics: MetricsTable
    segments: list[Segment]
    treated_frac: list[float]
    baseline_cost: float
    budget: float
    treated_cost: float
    incremental_gmv: float
    roi: float
    segment_cost_share: dict[str, float]
    segment_increment_share: dict[str, float]


def _knapsack_fractional(
    indices: Sequence[int],
    *,
    metrics: MetricsTable,
    budget: float,
    allow_negative_ite: bool,
    treated_frac: list[float],
) -> float:
    """
    Fractional knapsack (greedy by ite/cost).
    Mutates `treated_frac` for provided indices.
    Returns remaining budget.
    """
    remaining = float(budget)
    if remaining <= 0:
        return 0.0

    # Prepare candidates with feasible costs
    candidates: list[tuple[int, float]] = []
    for i in indices:
        c = metrics.cost[i]
        if c <= 0:
            continue
        ite = metrics.ite[i]
        if (not allow_negative_ite) and ite <= 0:
            continue
        ratio = ite / c
        candidates.append((i, ratio))

    # Sort by ratio descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    eps = 1e-12
    for i, _ratio in candidates:
        if remaining <= eps:
            break
        c = metrics.cost[i]
        if c <= remaining + eps:
            treated_frac[i] = 1.0
            remaining -= c
        else:
            treated_frac[i] = remaining / c
            remaining = 0.0
            break

    return remaining


def optimize_budget(
    metrics: MetricsTable,
    segments: list[Segment],
    *,
    budget_reduction_pct: float = 10.0,
    addict_accept_cost_share: float = 0.35,
    ite_blocking_mode: Literal["block_negative", "allow_negative"] = "block_negative",
) -> OptimizationResult:
    baseline_cost = float(sum(metrics.cost))
    budget = baseline_cost * (1.0 - float(budget_reduction_pct) / 100.0)
    budget = max(0.0, float(budget))

    allow_negative_ite = ite_blocking_mode == "allow_negative"

    n = len(metrics)
    treated_frac = [0.0] * n

    # Segment index sets
    gold_idx = [i for i, s in enumerate(segments) if s == "Gold"]
    addict_idx = [i for i, s in enumerate(segments) if s == "Addict"]
    # organic_idx / sinking_idx intentionally excluded from treatment in this simplified strategy

    # 1) Gold allocation
    remaining = budget
    remaining = _knapsack_fractional(
        gold_idx,
        metrics=metrics,
        budget=remaining,
        allow_negative_ite=allow_negative_ite,
        treated_frac=treated_frac,
    )

    # 2) Addict allocation with threshold receding (choose top by ratio then cap by cost share)
    if addict_idx and remaining > 0 and addict_accept_cost_share < 1.0:
        # Sort addict by ratio first
        candidates: list[tuple[int, float]] = []
        for i in addict_idx:
            c = metrics.cost[i]
            if c <= 0:
                continue
            ite = metrics.ite[i]
            if (not allow_negative_ite) and ite <= 0:
                continue
            candidates.append((i, ite / c))
        candidates.sort(key=lambda x: x[1], reverse=True)

        total_cost = sum(metrics.cost[i] for i, _ in candidates)
        keep: list[int] = []
        cum_cost = 0.0
        if total_cost > 1e-12:
            for i, _ in candidates:
                if (cum_cost / total_cost) <= float(addict_accept_cost_share):
                    keep.append(i)
                    cum_cost += metrics.cost[i]
                if (cum_cost / total_cost) > float(addict_accept_cost_share) and keep:
                    break
        if not keep and candidates:
            keep = [candidates[0][0]]
        addict_idx = keep

    remaining = _knapsack_fractional(
        addict_idx,
        metrics=metrics,
        budget=remaining,
        allow_negative_ite=allow_negative_ite,
        treated_frac=treated_frac,
    )

    treated_cost = 0.0
    incremental_gmv = 0.0
    for i in range(n):
        f = treated_frac[i]
        if f <= 0:
            continue
        treated_cost += metrics.cost[i] * f
        incremental_gmv += metrics.ite[i] * f

    roi = incremental_gmv / treated_cost if treated_cost > 1e-12 else 0.0

    seg_cost: dict[str, float] = {s: 0.0 for s in ["Gold", "Addict", "Organic", "Sinking"]}
    seg_inc: dict[str, float] = {s: 0.0 for s in ["Gold", "Addict", "Organic", "Sinking"]}
    for i, seg in enumerate(segments):
        f = treated_frac[i]
        if f <= 0:
            continue
        seg_cost[seg] += metrics.cost[i] * f
        seg_inc[seg] += metrics.ite[i] * f

    total_treated_cost = max(1e-12, treated_cost)
    total_inc = max(1e-12, incremental_gmv)
    segment_cost_share = {k: seg_cost[k] / total_treated_cost for k in seg_cost}
    segment_increment_share = {k: seg_inc[k] / total_inc for k in seg_inc}

    return OptimizationResult(
        metrics=metrics,
        segments=segments,
        treated_frac=treated_frac,
        baseline_cost=baseline_cost,
        budget=budget,
        treated_cost=treated_cost,
        incremental_gmv=incremental_gmv,
        roi=roi,
        segment_cost_share=segment_cost_share,
        segment_increment_share=segment_increment_share,
    )

