# src/evaluation.py
from typing import List, Tuple

import numpy as np

from src.catalog import item_catalog, item_sensitivity, get_item_category_dict
from src.policies import risk_score_full, rel_score


def eval_logs(logs):
    rels, risks = [], []
    hi_sens_cnt, total_cnt = 0, 0

    for entry in logs:
        state = entry["state"]
        items = entry["rec_items"]
        rel_vals = [rel_score(state, it) for it in items]
        risk_vals = [risk_score_full(state, it) for it in items]
        rels.append(np.mean(rel_vals))
        risks.append(np.mean(risk_vals))

        for it in items:
            if item_sensitivity(it) >= 0.7:
                hi_sens_cnt += 1
            total_cnt += 1

    return {
        "MeanRel": float(np.mean(rels)),
        "MeanRisk": float(np.mean(risks)),
        "HighSensRatio": hi_sens_cnt / total_cnt if total_cnt > 0 else 0.0,
    }


def compute_mean_jaccard(logs):
    if len(logs) <= 1:
        return 0.0
    js = []
    prev = set(logs[0]["rec_items"])
    for entry in logs[1:]:
        cur = set(entry["rec_items"])
        inter = len(prev & cur)
        union = len(prev | cur)
        js.append(inter / union if union > 0 else 0.0)
        prev = cur
    return float(np.mean(js))


def compute_category_diversity_from_logs(logs, top_k: int = 3):
    item_cat = get_item_category_dict()
    vals = []
    for entry in logs:
        cats = set(item_cat[it] for it in entry["rec_items"])
        vals.append(len(cats) / float(top_k))
    return float(np.mean(vals)) if vals else 0.0


def compute_temporal_diversity_from_logs(logs):
    if len(logs) <= 1:
        return 0.0
    vals = []
    prev = set(logs[0]["rec_items"])
    for entry in logs[1:]:
        cur = set(entry["rec_items"])
        inter = len(prev & cur)
        union = len(prev | cur)
        j = inter / union if union > 0 else 0.0
        vals.append(1.0 - j)
        prev = cur
    return float(np.mean(vals))


def bootstrap_ci(values: List[float], n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.5, 0.5

    boot_means = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_means.append(arr[idx].mean())

    lower = np.percentile(boot_means, 100 * (alpha / 2.0))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2.0))
    return float(lower), float(upper)


def estimate_item_distribution_by_state(logs, positive_state: int = 2):
    num_items = len(item_catalog)
    cnt_pos = np.zeros(num_items, dtype=np.float64)
    cnt_neg = np.zeros(num_items, dtype=np.float64)

    for entry in logs:
        s = entry["state"]
        items = entry["rec_items"]
        if s == positive_state:
            for it in items:
                cnt_pos[it] += 1
        else:
            for it in items:
                cnt_neg[it] += 1

    p1 = (cnt_pos + 1.0) / max(1.0, (cnt_pos + 1.0).sum())
    p0 = (cnt_neg + 1.0) / max(1.0, (cnt_neg + 1.0).sum())
    return p0, p1


def estimate_global_item_distribution_from_logs(logs):
    num_items = len(item_catalog)
    cnt = np.zeros(num_items, dtype=np.float64)
    for entry in logs:
        for it in entry["rec_items"]:
            cnt[it] += 1.0
    return (cnt + 1.0) / (cnt.sum() + num_items)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return 0.5 * np.abs(p - q).sum()