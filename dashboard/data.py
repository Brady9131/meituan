from __future__ import annotations

import csv
import io
import math
import random
from dataclasses import dataclass
from typing import Optional


REQUIRED_COLS = {"user_id", "ite", "pae", "cost"}

@dataclass(frozen=True)
class MetricsTable:
    user_id: list[int]
    ite: list[float]
    pae: list[float]
    cost: list[float]

    def __len__(self) -> int:
        return len(self.user_id)


def _to_float(x: str) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _try_read_csv(raw: bytes) -> MetricsTable:
    """
    Parse CSV bytes into an in-memory metrics table.

    NOTE: We intentionally avoid pandas/numpy to prevent native-library crashes.
    """
    text = raw.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    missing_cols = REQUIRED_COLS - set(reader.fieldnames or [])
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}. Got: {reader.fieldnames}")

    user_id: list[int] = []
    ite: list[float] = []
    pae: list[float] = []
    cost: list[float] = []

    for row in reader:
        uid_raw = row.get("user_id", "")
        try:
            uid = int(uid_raw)
        except Exception:
            # If user_id isn't numeric, fall back to a stable hash-like index.
            uid = abs(hash(uid_raw)) % (10**12)

        v_ite = _to_float(str(row.get("ite", "")))
        v_pae = _to_float(str(row.get("pae", "")))
        v_cost = _to_float(str(row.get("cost", "")))

        if v_ite is None or v_pae is None or v_cost is None:
            continue

        user_id.append(uid)
        ite.append(v_ite)
        pae.append(v_pae)
        cost.append(v_cost)

    return MetricsTable(user_id=user_id, ite=ite, pae=pae, cost=cost)


def load_user_metrics(uploaded_file, *, expected_cols: Optional[set[str]] = None) -> MetricsTable:
    """
    Load a per-user (or per-segment) metrics table.

    Expected columns (defaults): user_id, ite, pae, cost
    """
    raw = uploaded_file.read()

    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".parquet") or getattr(uploaded_file, "type", "").endswith("parquet"):
        # In some environments, parquet backends (e.g. pyarrow) may crash the Python process.
        # To keep the dashboard stable, we currently support CSV only.
        raise ValueError("Parquet format is currently not supported. Please convert to CSV first.")

    _ = expected_cols  # kept for compatibility; CSV parser validates required columns
    return _try_read_csv(raw)


def make_sample_data(n: int = 2000, seed: int = 42) -> MetricsTable:
    """
    Create a small synthetic dataset for dashboard preview.

    This is NOT the competition data; it's only for UI testing.
    """
    rng = random.Random(seed)
    user_id = [i for i in range(1, n + 1)]

    pae: list[float] = []
    ite: list[float] = []
    cost: list[float] = []

    for _ in range(n):
        p = rng.gauss(0.0, 1.0)
        base_gain = rng.gauss(0.8, 1.2)
        eps = rng.gauss(0.0, 0.35)
        g = base_gain - 0.35 * max(p, 0.0) + eps
        c = rng.gammavariate(2.0, 3.0)

        # Add some sparsity for "Organic" (low/zero incremental)
        if rng.random() < 0.25:
            g *= rng.uniform(0.0, 0.15)
            p = rng.gauss(-0.2, 0.7)

        pae.append(float(p))
        ite.append(float(g))
        cost.append(float(c))

    return MetricsTable(user_id=user_id, ite=ite, pae=pae, cost=cost)

