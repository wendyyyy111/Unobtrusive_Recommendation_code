# src/exposure.py
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from src.catalog import item_catalog, item_sensitivity

ITEM_EXPOSURE: Dict[int, float] = {}


def collect_exposure_training_data(
    states: np.ndarray,
    rec_logs: List[dict],
    sensitive_states: List[int],
) -> pd.DataFrame:
    cnt_state_item = defaultdict(int)
    cnt_state = Counter()

    for entry in rec_logs:
        s = int(entry["state"])
        items = entry["rec_items"]
        cnt_state[s] += 1
        for it in items:
            cnt_state_item[(s, it)] += 1

    rows = []
    for (s, item), c in cnt_state_item.items():
        total = cnt_state[s]
        p = c / total if total > 0 else 0.0
        rows.append({
            "state": s,
            "item": item,
            "count": c,
            "total_in_state": total,
            "p_item_given_state": p,
            "is_sensitive_state": 1 if s in sensitive_states else 0,
        })
    return pd.DataFrame(rows)


def compute_item_exposure_from_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Exposure stats DataFrame is empty.")

    df_item = (
        df.groupby(["item", "is_sensitive_state"])
        .agg(count=("count", "sum"))
        .reset_index()
    )

    sens = df_item[df_item["is_sensitive_state"] == 1].set_index("item")["count"]
    nonsens = df_item[df_item["is_sensitive_state"] == 0].set_index("item")["count"]

    all_items = set(sens.index).union(set(nonsens.index))
    rows = []
    for item in sorted(all_items):
        c_s = sens.get(item, 0)
        c_ns = nonsens.get(item, 0)
        p_s = (c_s + 1) / (c_s + c_ns + 2)
        p_ns = (c_ns + 1) / (c_s + c_ns + 2)
        odds = p_s / p_ns
        log_odds = np.log(odds + 1e-8)
        exposure_score = 1.0 / (1.0 + np.exp(-log_odds))
        rows.append({
            "item": item,
            "sens_count": c_s,
            "nonsens_count": c_ns,
            "exposure_score": exposure_score,
        })
    return pd.DataFrame(rows)


def load_item_exposure(df_item_exposure: pd.DataFrame):
    global ITEM_EXPOSURE
    ITEM_EXPOSURE = dict(
        zip(df_item_exposure["item"].astype(int), df_item_exposure["exposure_score"])
    )


def learned_inference_exposure(item_id: int) -> float:
    if not ITEM_EXPOSURE:
        return 0.5
    return float(ITEM_EXPOSURE.get(item_id, 0.5))


def set_item_exposure_mode(mode: str, df_exp_train: pd.DataFrame = None, random_seed: int = 42):
    global ITEM_EXPOSURE
    all_item_ids = [item["id"] for item in item_catalog]

    if mode == "data_driven":
        if df_exp_train is None:
            raise ValueError("df_exp_train is required for mode='data_driven'.")
        df_item_exposure = compute_item_exposure_from_stats(df_exp_train)
        load_item_exposure(df_item_exposure)
        return

    if mode == "sens_as_exposure":
        ITEM_EXPOSURE = {i: float(item_sensitivity(i)) for i in all_item_ids}

    elif mode == "freq_only":
        if df_exp_train is None:
            raise ValueError("df_exp_train is required for mode='freq_only'.")
        df_sens = df_exp_train[df_exp_train["is_sensitive_state"] == 1]
        if df_sens.empty:
            ITEM_EXPOSURE = {i: 0.5 for i in all_item_ids}
        else:
            cnt = df_sens.groupby("item")["count"].sum()
            max_cnt = cnt.max() if not cnt.empty else 1.0
            ITEM_EXPOSURE = {i: float(cnt.get(i, 0)) / max_cnt if max_cnt > 0 else 0.0 for i in all_item_ids}

    elif mode == "random":
        rng = np.random.RandomState(random_seed)
        ITEM_EXPOSURE = {i: float(rng.rand()) for i in all_item_ids}

    elif mode == "uniform":
        ITEM_EXPOSURE = {i: 0.5 for i in all_item_ids}

    else:
        raise ValueError(f"Unknown exposure mode: {mode}")