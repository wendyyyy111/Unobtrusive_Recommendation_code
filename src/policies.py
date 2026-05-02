# src/policies.py
from typing import Any, Dict, List

import numpy as np

from src.catalog import (
    item_catalog,
    rel_score,
    item_sensitivity,
    signal_privacy,
    comfort_penalty,
)
from src.exposure import learned_inference_exposure
from src.config import TOP_K


def risk_components(state: int, item_id: int) -> Dict[str, float]:
    return {
        "item_sens": item_sensitivity(item_id),
        "state_privacy": signal_privacy(state),
        "exposure": learned_inference_exposure(item_id),
        "comfort": comfort_penalty(state, item_id),
    }


def aggregate_risk_from_components(
    comps: Dict[str, float],
    use_item_sens: bool = True,
    use_state_privacy: bool = True,
    use_exposure: bool = True,
):
    r = 0.0
    if use_item_sens:
        r += comps["item_sens"]
    if use_state_privacy:
        r += comps["state_privacy"]
    if use_exposure:
        r += comps["exposure"]
    return r


def risk_score_full(state, item_id, w_sens: float = 0.4, w_signal: float = 0.2, w_exposure: float = 0.4):
    cs = item_sensitivity(item_id)
    sp = signal_privacy(state)
    ie = learned_inference_exposure(item_id)
    return w_sens * cs + w_signal * sp + w_exposure * ie


def risk_score_simple(state, item_id):
    return 0.5 * item_sensitivity(item_id)


def baseline_ranking(state, top_k: int = TOP_K):
    scores = [(item["id"], rel_score(state, item["id"])) for item in item_catalog]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def simulate_baseline_logs(states, top_k: int = TOP_K):
    logs = []
    for t, s in enumerate(states):
        ranked = baseline_ranking(int(s), top_k=top_k)
        logs.append({"t": t, "state": int(s), "rec_items": [it for it, _ in ranked]})
    return logs


def fixed_filter_ranking(state, top_k: int = TOP_K):
    allowed_ids = [0, 1, 2]
    scores = [(i, rel_score(state, i)) for i in allowed_ids]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def simulate_fixed_filter_logs(states, top_k: int = TOP_K):
    logs = []
    for t, s in enumerate(states):
        ranked = fixed_filter_ranking(int(s), top_k=top_k)
        logs.append({"t": t, "state": int(s), "rec_items": [it for it, _ in ranked]})
    return logs


def dp_noisy_ranking(state: int, epsilon: float = 1.0, top_k: int = TOP_K):
    scores = []
    scale = 1.0 / max(epsilon, 1e-6)
    for item in item_catalog:
        i = item["id"]
        noise = np.random.laplace(loc=0.0, scale=scale)
        scores.append((i, rel_score(state, i) + noise))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def simulate_dp_noisy_logs(states, epsilon: float = 1.0, top_k: int = TOP_K):
    logs = []
    for t, s in enumerate(states):
        ranked = dp_noisy_ranking(int(s), epsilon=epsilon, top_k=top_k)
        logs.append({"t": t, "state": int(s), "rec_items": [it for it, _ in ranked]})
    return logs


def riskonly_itemsens_score(state: int, item_id: int, lam_item: float = 0.5):
    return rel_score(state, item_id) - lam_item * item_sensitivity(item_id)


def riskonly_itemsens_ranking(state: int, lam_item: float = 0.5, top_k: int = TOP_K):
    scores = [(item["id"], riskonly_itemsens_score(state, item["id"], lam_item)) for item in item_catalog]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [it for it, _ in scores[:top_k]]


def simulate_riskonly_itemsens_logs(states, lam_item: float = 0.5, top_k: int = TOP_K):
    logs = []
    for t, s in enumerate(states):
        rec_items = riskonly_itemsens_ranking(int(s), lam_item=lam_item, top_k=top_k)
        logs.append({"t": t, "state": int(s), "rec_items": rec_items})
    return logs


def risk_aware_score(
    state: int,
    item_id: int,
    lam: float = 0.5,
    mu: float = 0.5,
    use_full_risk: bool = True,
    use_item_sens: bool = True,
    use_state_privacy: bool = True,
    use_exposure: bool = True,
    use_comfort: bool = True,
):
    r = rel_score(state, item_id)

    if use_full_risk:
        comps = risk_components(state, item_id)
        privacy_risk = aggregate_risk_from_components(
            comps,
            use_item_sens=use_item_sens,
            use_state_privacy=use_state_privacy,
            use_exposure=use_exposure,
        )
        comfort = comps["comfort"] if use_comfort else 0.0
        return r - lam * privacy_risk - mu * comfort

    rk = risk_score_simple(state, item_id)
    cp = comfort_penalty(state, item_id) if use_comfort else 0.0
    return r - lam * rk - mu * cp


def risk_aware_ranking(
    state,
    lam=0.5,
    mu=0.5,
    top_k: int = TOP_K,
    use_full_risk: bool = True,
    use_item_sens: bool = True,
    use_state_privacy: bool = True,
    use_exposure: bool = True,
    use_comfort: bool = True,
):
    scores = []
    for item in item_catalog:
        s_val = risk_aware_score(
            state, item["id"],
            lam=lam, mu=mu,
            use_full_risk=use_full_risk,
            use_item_sens=use_item_sens,
            use_state_privacy=use_state_privacy,
            use_exposure=use_exposure,
            use_comfort=use_comfort,
        )
        scores.append((item["id"], s_val))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def simulate_risk_aware_logs(
    states,
    lam=0.5,
    mu=0.5,
    top_k: int = TOP_K,
    use_full_risk: bool = True,
    use_item_sens: bool = True,
    use_state_privacy: bool = True,
    use_exposure: bool = True,
    use_comfort: bool = True,
):
    logs = []
    for t, s in enumerate(states):
        ranked = risk_aware_ranking(
            int(s),
            lam=lam,
            mu=mu,
            top_k=top_k,
            use_full_risk=use_full_risk,
            use_item_sens=use_item_sens,
            use_state_privacy=use_state_privacy,
            use_exposure=use_exposure,
            use_comfort=use_comfort,
        )
        logs.append({"t": t, "state": int(s), "rec_items": [it for it, _ in ranked]})
    return logs


def smooth_ranking(prev_hist, state, lam=0.5, mu=0.5, alpha=0.4, top_k: int = TOP_K, use_full_risk: bool = True):
    scores = []
    for item in item_catalog:
        i = item["id"]
        base_score = risk_aware_score(state, i, lam=lam, mu=mu, use_full_risk=use_full_risk)
        hist_score = 0.0 if prev_hist is None else float(prev_hist[i])
        s_val = (1.0 - alpha) * base_score + alpha * hist_score
        scores.append((i, s_val))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def simulate_risk_aware_logs_smooth(
    states, lam=0.5, mu=0.5, alpha=0.4, top_k: int = TOP_K,
    use_full_risk: bool = True, decay: float = 0.6, reward: float = 0.05,
):
    num_items = len(item_catalog)
    hist = np.ones(num_items, dtype=np.float32) / float(num_items)
    logs = []

    for t, s in enumerate(states):
        ranked = smooth_ranking(hist, int(s), lam=lam, mu=mu, alpha=alpha, top_k=top_k, use_full_risk=use_full_risk)
        rec_items = [it for it, _ in ranked]

        hist *= decay
        for it in rec_items:
            hist[it] += reward
        total = float(hist.sum())
        if total > 0:
            hist /= total

        logs.append({"t": t, "state": int(s), "rec_items": rec_items})
    return logs


def is_high_sens_item(item_id: int, threshold: float = 0.7):
    return item_sensitivity(item_id) >= threshold


def sensitive_cap_ranking(state: int, sens_cap: float = 0.2, hist_high_sens_ratio: float = 0.0, top_k: int = TOP_K):
    if hist_high_sens_ratio >= sens_cap:
        candidate_items = [it for it in item_catalog if not is_high_sens_item(it["id"])]
    else:
        candidate_items = item_catalog

    scores = [(item["id"], rel_score(state, item["id"])) for item in candidate_items]
    if not scores:
        return [it for it, _ in baseline_ranking(state, top_k=top_k)]

    scores.sort(key=lambda x: x[1], reverse=True)
    return [it for it, _ in scores[:top_k]]


def simulate_sensitive_cap_logs(states, sens_cap: float = 0.2, top_k: int = TOP_K):
    logs = []
    high_sens_count = 0
    total_count = 0

    for t, s in enumerate(states):
        hist_ratio = (high_sens_count / total_count) if total_count > 0 else 0.0
        rec_items = sensitive_cap_ranking(int(s), sens_cap=sens_cap, hist_high_sens_ratio=hist_ratio, top_k=top_k)

        for it in rec_items:
            if is_high_sens_item(it):
                high_sens_count += 1
            total_count += 1

        logs.append({"t": t, "state": int(s), "rec_items": rec_items})
    return logs


def estimate_global_popularity_from_baseline(states, top_k: int = TOP_K, pool_size: int = 3):
    from collections import Counter

    logs = simulate_baseline_logs(states, top_k=top_k)
    cnt = Counter()
    for entry in logs:
        for it in entry["rec_items"]:
            cnt[it] += 1
    return [it for it, _ in cnt.most_common(pool_size)]


def popular_mix_ranking(state: int, popular_pool: List[int], mix_bonus: float = 0.3, top_k: int = TOP_K):
    popular_set = set(popular_pool)
    scores = []
    for item in item_catalog:
        i = item["id"]
        bonus = mix_bonus if i in popular_set else 0.0
        scores.append((i, rel_score(state, i) + bonus))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [it for it, _ in scores[:top_k]]


def simulate_popular_mix_logs(states, popular_pool: List[int], mix_bonus: float = 0.3, top_k: int = TOP_K):
    logs = []
    for t, s in enumerate(states):
        rec_items = popular_mix_ranking(int(s), popular_pool=popular_pool, mix_bonus=mix_bonus, top_k=top_k)
        logs.append({"t": t, "state": int(s), "rec_items": rec_items})
    return logs