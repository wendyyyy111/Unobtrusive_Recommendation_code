# src/experiments.py
from itertools import product
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from src.attackers import (
    train_and_eval_multi_attackers,
    train_and_eval_strong_attackers,
)
from src.catalog import item_catalog
from src.config import TOP_K
from src.evaluation import (
    bootstrap_ci,
    compute_category_diversity_from_logs,
    compute_mean_jaccard,
    compute_temporal_diversity_from_logs,
    estimate_global_item_distribution_from_logs,
    estimate_item_distribution_by_state,
    eval_logs,
    total_variation_distance,
)
from src.exposure import set_item_exposure_mode
from src.policies import (
    estimate_global_popularity_from_baseline,
    popular_mix_ranking,
    risk_aware_score,
    simulate_baseline_logs,
    simulate_dp_noisy_logs,
    simulate_fixed_filter_logs,
    simulate_popular_mix_logs,
    simulate_risk_aware_logs,
    simulate_risk_aware_logs_smooth,
    simulate_riskonly_itemsens_logs,
    simulate_sensitive_cap_logs,
)
from src.utils import set_global_seed


def simulate_logs_for_setting(
    states: np.ndarray,
    lam: float,
    mu: float,
    alpha: float,
    top_k: int = TOP_K,
    use_smooth: bool = True,
    use_full_risk: bool = True,
):
    if lam == 0.0 and mu == 0.0:
        return simulate_baseline_logs(states, top_k=top_k)

    if use_smooth:
        return simulate_risk_aware_logs_smooth(
            states, lam=lam, mu=mu, alpha=alpha, top_k=top_k,
            use_full_risk=use_full_risk
        )

    return simulate_risk_aware_logs(
        states, lam=lam, mu=mu, top_k=top_k,
        use_full_risk=use_full_risk
    )


def eval_setting(
    states: np.ndarray,
    lam: float,
    mu: float,
    alpha: float,
    top_k: int = TOP_K,
    use_smooth: bool = True,
    use_full_risk: bool = True,
    drop_prob: float = 0.3,
):
    logs = simulate_logs_for_setting(
        states=states, lam=lam, mu=mu, alpha=alpha,
        top_k=top_k, use_smooth=use_smooth, use_full_risk=use_full_risk
    )
    metrics = eval_logs(logs)
    mean_jaccard = compute_mean_jaccard(logs)

    attack_res = train_and_eval_strong_attackers(logs, drop_prob=drop_prob, L=5, seed=42)
    auc_best = attack_res["GlobalBestAUC"]

    return {
        "lambda": lam,
        "alpha": alpha,
        "mu": mu,
        "MeanRel": metrics["MeanRel"],
        "HighSensRatio": metrics["HighSensRatio"],
        "MeanRisk": metrics["MeanRisk"],
        "MeanJaccard": mean_jaccard,
        "AttackAUC": auc_best,
    }


def eval_setting_with_seeds(
    states: np.ndarray,
    lam: float,
    mu: float,
    alpha: float,
    seeds: List[int],
    top_k: int = TOP_K,
    use_smooth: bool = True,
    use_full_risk: bool = True,
    drop_prob: float = 0.3,
):
    rel_list, hs_list, risk_list, jacc_list, auc_list = [], [], [], [], []

    for seed in seeds:
        set_global_seed(seed)

        logs = simulate_logs_for_setting(
            states, lam=lam, mu=mu, alpha=alpha,
            top_k=top_k, use_smooth=use_smooth, use_full_risk=use_full_risk
        )
        metrics = eval_logs(logs)
        jacc = compute_mean_jaccard(logs)

        attack_res = train_and_eval_strong_attackers(logs, drop_prob=drop_prob, L=5, seed=seed)
        auc_best = attack_res["GlobalBestAUC"]

        rel_list.append(metrics["MeanRel"])
        hs_list.append(metrics["HighSensRatio"])
        risk_list.append(metrics["MeanRisk"])
        jacc_list.append(jacc)
        auc_list.append(auc_best)

    ci_low, ci_high = bootstrap_ci(auc_list, n_boot=10000, alpha=0.05)

    return {
        "lambda": lam,
        "mu": mu,
        "alpha": alpha,
        "MeanRel": float(np.mean(rel_list)),
        "HighSensRatio": float(np.mean(hs_list)),
        "MeanRisk": float(np.mean(risk_list)),
        "MeanJaccard": float(np.mean(jacc_list)),
        "AttackAUC": float(np.mean(auc_list)),
        "AttackAUC_CI_low": ci_low,
        "AttackAUC_CI_high": ci_high,
        "AUC_values": auc_list,
    }


def run_lambda_sweep_with_seeds(
    states: np.ndarray,
    lambda_list: List[float],
    fixed_alpha: float = 0.4,
    mu: float = 0.5,
    seeds: List[int] = None,
    use_smooth: bool = True,
    use_full_risk: bool = True,
    drop_prob: float = 0.3,
):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    results = []
    for lam in lambda_list:
        res = eval_setting_with_seeds(
            states=states,
            lam=lam,
            mu=mu,
            alpha=fixed_alpha,
            seeds=seeds,
            top_k=TOP_K,
            use_smooth=use_smooth,
            use_full_risk=use_full_risk,
            drop_prob=drop_prob,
        )
        results.append(res)
    return pd.DataFrame(results).sort_values("lambda").reset_index(drop=True)


def run_baseline_comparison(states: np.ndarray, drop_prob: float = 0.3, seed: int = 42):
    methods = []

    methods.append({"name": "Baseline-R", "logs": simulate_baseline_logs(states, top_k=TOP_K)})
    methods.append({"name": "Fixed-Filter", "logs": simulate_fixed_filter_logs(states, top_k=TOP_K)})
    methods.append({"name": "Sensitive-Cap(0.2)", "logs": simulate_sensitive_cap_logs(states, sens_cap=0.2, top_k=TOP_K)})
    methods.append({"name": "DP-Noisy(eps=0.5)", "logs": simulate_dp_noisy_logs(states, epsilon=0.5, top_k=TOP_K)})

    popular_pool = estimate_global_popularity_from_baseline(states, top_k=TOP_K, pool_size=3)
    methods.append({
        "name": "Popular-Mix(bonus=0.3)",
        "logs": simulate_popular_mix_logs(states, popular_pool=popular_pool, mix_bonus=0.3, top_k=TOP_K),
    })

    methods.append({
        "name": "RiskOnly-ItemSens(lam_item=0.5)",
        "logs": simulate_riskonly_itemsens_logs(states, lam_item=0.5, top_k=TOP_K),
    })

    methods.append({
        "name": "RA(lambda=0.5)",
        "logs": simulate_risk_aware_logs(states, lam=0.5, mu=0.0, top_k=TOP_K, use_full_risk=True),
    })
    methods.append({
        "name": "RA+Smooth(lambda=0.5, alpha=0.4)",
        "logs": simulate_risk_aware_logs_smooth(states, lam=0.5, mu=0.0, alpha=0.4, top_k=TOP_K, use_full_risk=True),
    })
    methods.append({
        "name": "RA+Smooth+Comfort(lambda=0.5, mu=0.5, alpha=0.4)",
        "logs": simulate_risk_aware_logs_smooth(states, lam=0.5, mu=0.5, alpha=0.4, top_k=TOP_K, use_full_risk=True),
    })

    rows = []
    for m in methods:
        logs = m["logs"]
        metrics = eval_logs(logs)
        mean_jaccard = compute_mean_jaccard(logs)
        cat_div = compute_category_diversity_from_logs(logs, top_k=TOP_K)
        temp_div = compute_temporal_diversity_from_logs(logs)
        attack_res = train_and_eval_strong_attackers(logs, drop_prob=drop_prob, L=5, seed=seed)

        rows.append({
            "Method": m["name"],
            "MeanRel": metrics["MeanRel"],
            "MeanRisk": metrics["MeanRisk"],
            "HighSensRatio": metrics["HighSensRatio"],
            "MeanJaccard": mean_jaccard,
            "CategoryDiversity": cat_div,
            "TemporalDiversity": temp_div,
            "AttackAUC": attack_res["GlobalBestAUC"],
        })

    return pd.DataFrame(rows)


def run_baseline_comparison_multi_seeds(states: np.ndarray, drop_prob: float = 0.3, seeds: List[int] = None):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    per_seed_dfs = []
    for sd in seeds:
        set_global_seed(sd)
        df_one = run_baseline_comparison(states, drop_prob=drop_prob, seed=sd)
        df_one["seed"] = sd
        per_seed_dfs.append(df_one)

    df_all = pd.concat(per_seed_dfs, ignore_index=True)

    agg_rows = []
    for method, g in df_all.groupby("Method"):
        row = {"Method": method}
        for col in ["MeanRel", "MeanRisk", "HighSensRatio", "MeanJaccard", "CategoryDiversity", "TemporalDiversity", "AttackAUC"]:
            vals = g[col].values.astype(float)
            row[col + "_mean"] = float(vals.mean())
            row[col + "_std"] = float(vals.std(ddof=0))
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def theory_smooth_ranking(state: int, base_scores: Dict[int, float], q: np.ndarray, alpha: float, top_k: int = TOP_K):
    scores = []
    for item in item_catalog:
        i = item["id"]
        s_val = (1.0 - alpha) * base_scores[i] + alpha * float(q[i])
        scores.append((i, s_val))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [it for it, _ in scores[:top_k]]


def run_alpha_tv_auc_analysis_multi_q_multi_seeds(
    states: np.ndarray,
    lam: float = 0.5,
    mu: float = 0.5,
    alphas: List[float] = None,
    drop_prob: float = 0.3,
    positive_state: int = 2,
    q_mode: str = "avg_p0_p1",
    seeds: List[int] = None,
):
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.4, 0.6, 0.7]
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    logs_base = simulate_risk_aware_logs(states, lam=lam, mu=mu, top_k=TOP_K, use_full_risk=True)
    p0, p1 = estimate_item_distribution_by_state(logs_base, positive_state=positive_state)

    if q_mode == "avg_p0_p1":
        q = 0.5 * (p0 + p1)
    elif q_mode == "p0_only":
        q = p0
    elif q_mode == "baseline_global":
        baseline_logs = simulate_baseline_logs(states, top_k=TOP_K)
        q = estimate_global_item_distribution_from_logs(baseline_logs)
    else:
        raise ValueError(f"Unknown q_mode: {q_mode}")

    base_scores_all = []
    for entry in logs_base:
        s = entry["state"]
        scores_dict = {item["id"]: risk_aware_score(s, item["id"], lam=lam, mu=mu, use_full_risk=True) for item in item_catalog}
        base_scores_all.append(scores_dict)

    rows = []
    for alpha in alphas:
        tv_list, auc_list = [], []

        for sd in seeds:
            set_global_seed(sd)

            logs_theory = []
            for t, entry in enumerate(logs_base):
                s = entry["state"]
                rec_items = theory_smooth_ranking(
                    state=s,
                    base_scores=base_scores_all[t],
                    q=q,
                    alpha=alpha,
                    top_k=TOP_K,
                )
                logs_theory.append({"t": t, "state": int(s), "rec_items": rec_items})

            p0_alpha, p1_alpha = estimate_item_distribution_by_state(logs_theory, positive_state=positive_state)
            tv_alpha = total_variation_distance(p0_alpha, p1_alpha)
            tv_list.append(tv_alpha)

            attack_res = train_and_eval_strong_attackers(logs_theory, drop_prob=drop_prob, L=5, seed=sd)
            auc_list.append(attack_res["GlobalBestAUC"])

        tv_arr = np.array(tv_list, dtype=float)
        auc_arr = np.array(auc_list, dtype=float)
        auc_ci_low, auc_ci_high = bootstrap_ci(auc_arr, n_boot=10000, alpha=0.05)

        rows.append({
            "alpha": alpha,
            "TV_mean": float(tv_arr.mean()),
            "TV_std": float(tv_arr.std(ddof=0)),
            "AttackAUC_mean": float(auc_arr.mean()),
            "AttackAUC_CI_low": auc_ci_low,
            "AttackAUC_CI_high": auc_ci_high,
        })

    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)