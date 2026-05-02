# scripts/run_wesad_pipeline.py
from src.config import (
    ALL_SUBJECTS,
    DATA_DIR,
    DEFAULT_ALPHA,
    DEFAULT_DROP_PROB,
    DEFAULT_LAMBDA,
    DEFAULT_MU,
    DEFAULT_SEEDS,
    LAMBDA_LIST,
)
from src.data_wesad import build_feature_dataset
from src.exposure import (
    collect_exposure_training_data,
    compute_item_exposure_from_stats,
    load_item_exposure,
    set_item_exposure_mode,
)
from src.attackers import build_embedding_attack_dataset_from_wesad, train_and_eval_multi_attackers
from src.experiments import (
    run_alpha_tv_auc_analysis_multi_q_multi_seeds,
    run_baseline_comparison_multi_seeds,
    run_lambda_sweep_with_seeds,
)
from src.plotting import plot_lambda_sweep_with_errorbars, plot_tradeoff_curves_lambda
from src.policies import simulate_baseline_logs
from src.state_model import train_state_classifier_subject_split


def main():
    print("Building feature dataset from WESAD ...")
    X, y, meta = build_feature_dataset(ALL_SUBJECTS, DATA_DIR)
    print(f"Total windows: {len(y)}, feature dim = {X.shape[1]}")

    print("\n[Embedding-level Attack] ...")
    X_emb, y_emb = build_embedding_attack_dataset_from_wesad(X_feat=X, y_state=y, positive_states=[2])
    embed_attack_res = train_and_eval_multi_attackers(X_emb, y_emb)
    print(
        f"[Embedding-level Attack] Best AUC: {embed_attack_res['BestAUC']:.4f} "
        f"({embed_attack_res['BestModel']}) on {len(y_emb)} windows."
    )

    print("\nTraining state classifier with subject-level split ...")
    state_clf, y_test, y_pred_test, idx_test, subj_split = train_state_classifier_subject_split(X, y, meta, random_state=42)

    states_gt_all = y
    states_pred_all = state_clf.predict(X)

    states_gt = states_gt_all[idx_test]
    states_pred = states_pred_all[idx_test]

    state_sources = {
        "GT-States": states_gt,
        "Pred-States": states_pred,
    }

    drop_prob = DEFAULT_DROP_PROB

    for src_name, states in state_sources.items():
        print("\n======================================")
        print(f"Running pipeline with {src_name}")
        print("======================================")

        baseline_logs = simulate_baseline_logs(states)
        df_exp_train = collect_exposure_training_data(states=states, rec_logs=baseline_logs, sensitive_states=[2])
        df_item_exposure = compute_item_exposure_from_stats(df_exp_train)
        load_item_exposure(df_item_exposure)
        set_item_exposure_mode("data_driven", df_exp_train=df_exp_train)

        print(f"\n[STEP] Baseline comparison under {src_name} ...")
        df_baseline = run_baseline_comparison_multi_seeds(states, drop_prob=drop_prob, seeds=DEFAULT_SEEDS)
        print(df_baseline)
        df_baseline.to_csv(f"wesad_baseline_multi_seed_{src_name.replace('-', '').lower()}.csv", index=False)

        print(f"\n[STEP] Lambda sweep under {src_name} ...")
        df_lambda = run_lambda_sweep_with_seeds(
            states=states,
            lambda_list=LAMBDA_LIST,
            fixed_alpha=DEFAULT_ALPHA,
            mu=DEFAULT_MU,
            seeds=DEFAULT_SEEDS,
            use_smooth=True,
            use_full_risk=True,
            drop_prob=drop_prob,
        )
        print(df_lambda)
        df_lambda.to_csv(f"wesad_lambda_sweep_{src_name.replace('-', '').lower()}.csv", index=False)

        prefix = f"wesad_{src_name.replace('-', '').lower()}"
        plot_lambda_sweep_with_errorbars(df_lambda, save_prefix=prefix, show_rel=True)
        plot_tradeoff_curves_lambda(df_lambda, save_prefix=prefix + "_no_ci")

        if src_name == "GT-States":
            print(f"\n[STEP] Alpha-TV-AUC analysis under {src_name} ...")
            df_alpha_tv = run_alpha_tv_auc_analysis_multi_q_multi_seeds(
                states=states,
                lam=DEFAULT_LAMBDA,
                mu=DEFAULT_MU,
                alphas=[0.0, 0.1, 0.2, 0.4, 0.6, 0.7],
                drop_prob=drop_prob,
                positive_state=2,
                q_mode="avg_p0_p1",
                seeds=DEFAULT_SEEDS,
            )
            print(df_alpha_tv)
            df_alpha_tv.to_csv("wesad_alpha_tv_auc_gtstates.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()