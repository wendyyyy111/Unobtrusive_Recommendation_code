# src/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_attack_auc_vs_drop_prob(df_dp: pd.DataFrame, save_prefix: str = None):
    x = df_dp["drop_prob"].values
    y = df_dp["AttackAUC"].values
    plt.figure(figsize=(6, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("drop_prob")
    plt.ylabel("AttackAUC (↓)")
    plt.title("AttackAUC vs drop_prob")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_attackauc_vs_dropprob.png", dpi=300)
    plt.show()


def plot_tradeoff_curves_lambda(df_lambda: pd.DataFrame, save_prefix: str = None):
    x = df_lambda["lambda"].values
    plt.figure(figsize=(8, 6))
    plt.plot(x, df_lambda["MeanRel"].values, marker="o", label="Rel (↑)")
    plt.plot(x, df_lambda["HighSensRatio"].values, marker="s", label="HighSensRatio (↓)")
    plt.plot(x, df_lambda["MeanJaccard"].values, marker="^", label="MeanJaccard (↑)")
    plt.plot(x, df_lambda["AttackAUC"].values, marker="x", label="Best Attack AUC (↓)")
    plt.xlabel(r"$\lambda$ (Risk weight)")
    plt.ylabel("Metric value")
    plt.title("Trade-off w.r.t. $\\lambda$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_lambda_tradeoff.png", dpi=300)
    plt.show()


def plot_lambda_sweep_with_errorbars(df_lambda: pd.DataFrame, save_prefix: str = None, show_rel: bool = True):
    lambdas = df_lambda["lambda"].values
    auc_mean = df_lambda["AttackAUC"].values
    ci_low = df_lambda["AttackAUC_CI_low"].values
    ci_high = df_lambda["AttackAUC_CI_high"].values

    yerr_lower = auc_mean - ci_low
    yerr_upper = ci_high - auc_mean
    yerr = np.vstack([yerr_lower, yerr_upper])

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        lambdas, auc_mean, yerr=yerr, fmt="o-", capsize=4,
        label="Attack AUC (mean ± 95% CI)", color="tab:red",
    )

    if show_rel and "MeanRel" in df_lambda.columns:
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        rel_vals = df_lambda["MeanRel"].values
        ax2.plot(lambdas, rel_vals, marker="s", linestyle="--", color="tab:blue", label="MeanRel")
        ax2.set_ylabel("MeanRel (↑)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        plt.legend(loc="best")

    plt.xlabel(r"$\lambda$ (Risk weight)")
    plt.ylabel("Attack AUC (↓)", color="tab:red")
    plt.title("Lambda Sweep with 95% CI on Attack AUC")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_lambda_auc_ci.png", dpi=300)
    plt.show()