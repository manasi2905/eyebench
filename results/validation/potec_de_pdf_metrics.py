"""Summarize PoTeC-DE trial-level predictions into benchmark CSVs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("results/raw")
OUT_DIR = Path("results/eyebench_benchmark_results")
OUT_SUMMARY = OUT_DIR / "potec_de_pdf_ready_all_metrics.csv"
OUT_FOLDS = OUT_DIR / "potec_de_fold_level_all_metrics.csv"

TARGET_MODELS = {
    "LogisticRegressionMLArgs",
    "KNNMLArgs",
    "LRKNNEnsembleMLArgs",
}

REGIME_MAP = {
    "seen_subject_unseen_item": "Unseen text",
    "unseen_subject_seen_item": "Unseen reader",
    "unseen_subject_unseen_item": "Both unseen",
}


def model_name_from_path(path: Path) -> str:
    """Return the model name embedded in a raw-results path."""
    match = re.search(r"\+model=([^,\\]+)", str(path))
    return match.group(1) if match else "UNKNOWN_MODEL"


def parse_probability(value: object) -> float:
    """Normalize a stored prediction to the positive-class probability."""
    if isinstance(value, str):
        value = value.strip()
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return float(value)

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).squeeze()
        if arr.ndim == 0:
            return float(arr)
        if len(arr) > 1:
            return float(arr[1])
        return float(arr[0])

    return float(value)


def accuracy_score_np(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute accuracy without requiring scikit-learn."""
    true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    return float(np.mean(true == pred))


def balanced_accuracy_score_np(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute binary balanced accuracy: mean of sensitivity and specificity."""
    true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)

    recalls = []
    for label in [0, 1]:
        mask = true == label
        if mask.any():
            recalls.append(np.mean(pred[mask] == label))

    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def roc_auc_score_np(y_true: pd.Series, y_score: pd.Series) -> float:
    """Compute binary AUROC using average ranks for tied scores."""
    true = pd.Series(y_true).astype(int).reset_index(drop=True)
    score = pd.Series(y_score).astype(float).reset_index(drop=True)

    n_pos = int((true == 1).sum())
    n_neg = int((true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = score.rank(method="average")
    pos_rank_sum = ranks[true == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def collect_fold_metrics() -> pd.DataFrame:
    """Collect fold-level metrics for the PoTeC-DE models we keep."""
    rows = []

    for csv_path in RAW_DIR.rglob("trial_level_test_results.csv"):
        if "PoTeC_DE" not in str(csv_path):
            continue

        model = model_name_from_path(csv_path)
        if model not in TARGET_MODELS:
            continue

        df = pd.read_csv(csv_path)
        df = df[df["eval_type"] == "test"].copy()

        df["prediction_prob"] = df["prediction_prob"].apply(parse_probability)
        df["pred_label"] = (df["prediction_prob"] > 0.5).astype(int)

        for (regime, fold), group in df.groupby(["eval_regime", "fold_index"]):
            if regime not in REGIME_MAP:
                continue

            y_true = group["label"].astype(int)
            y_prob = group["prediction_prob"]
            y_pred = group["pred_label"]

            rows.append(
                {
                    "Model": model,
                    "Fold": int(fold),
                    "Regime": REGIME_MAP[regime],
                    "AUROC": roc_auc_score_np(y_true, y_prob),
                    "Accuracy": accuracy_score_np(y_true, y_pred),
                    "Balanced accuracy": balanced_accuracy_score_np(y_true, y_pred),
                    "N": len(group),
                    "Positives": int(y_true.sum()),
                }
            )

    return pd.DataFrame(rows)


def summarize_for_pdf(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Average fold metrics into the PDF summary table."""
    summary_rows = []

    for model, model_df in fold_df.groupby("Model"):
        row = {"Model": model}
        regime_aurocs = []

        for regime in ["Unseen text", "Unseen reader", "Both unseen"]:
            regime_df = model_df[model_df["Regime"] == regime]

            for metric in ["AUROC", "Accuracy", "Balanced accuracy"]:
                mean = regime_df[metric].mean()
                std = regime_df[metric].std(ddof=0)

                row[f"{regime} {metric}"] = round(mean, 3)
                row[f"{regime} {metric} +/- std"] = f"{mean:.3f} +/- {std:.3f}"

            regime_aurocs.append(regime_df["AUROC"].mean())

        row["Mean AUROC"] = round(float(np.mean(regime_aurocs)), 3)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values("Mean AUROC", ascending=False)


def main() -> None:
    fold_df = collect_fold_metrics()

    summary_df = summarize_for_pdf(fold_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    fold_df.to_csv(OUT_FOLDS, index=False)

    display_columns = [
        "Model",
        "Unseen text AUROC +/- std",
        "Unseen text Accuracy +/- std",
        "Unseen text Balanced accuracy +/- std",
        "Unseen reader AUROC +/- std",
        "Unseen reader Accuracy +/- std",
        "Unseen reader Balanced accuracy +/- std",
        "Both unseen AUROC +/- std",
        "Both unseen Accuracy +/- std",
        "Both unseen Balanced accuracy +/- std",
        "Mean AUROC",
    ]
    print(summary_df[display_columns].to_string(index=False))
    print(f"\nSaved summary to {OUT_SUMMARY}")
    print(f"Saved fold-level metrics to {OUT_FOLDS}")


if __name__ == "__main__":
    main()
