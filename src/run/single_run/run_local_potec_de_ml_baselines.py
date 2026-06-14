"""Local PoTeC-DE baseline tuning and rerun helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning_fabric as lf
import numpy as np
import pandas as pd
import torch

from src.configs.constants import ItemLevelFeaturesModes, SetNames
from src.configs.data import PoTeC_DE
from src.configs.models.ml.KNNeighbors import KNNMLArgs
from src.configs.models.ml.LRKNNEnsemble import LRKNNEnsembleMLArgs
from src.configs.models.ml.LogisticRegression import LogisticRegressionMLArgs
from src.configs.trainers import TrainerML
from src.run.single_run.test_ml import process_single_run


OUT_DIR = Path("results/student_runs/potec_de_local_tuning")
TUNING_LOG_PATH = OUT_DIR / "validation_tuning_log.csv"

REGIMES = [
    SetNames.SEEN_SUBJECT_UNSEEN_ITEM,
    SetNames.UNSEEN_SUBJECT_SEEN_ITEM,
    SetNames.UNSEEN_SUBJECT_UNSEEN_ITEM,
]


def roc_auc_score_binary(y_true: pd.Series, y_score: pd.Series) -> float:
    """Binary AUROC with average ranks for ties."""
    true = pd.Series(y_true).astype(int).reset_index(drop=True)
    score = pd.Series(y_score).apply(positive_class_score).astype(float).reset_index(drop=True)

    n_pos = int((true == 1).sum())
    n_neg = int((true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = score.rank(method="average")
    pos_rank_sum = ranks[true == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def positive_class_score(value: object) -> float:
    """Convert scalar/list prediction output to a positive-class score."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).squeeze()
        if arr.ndim == 0:
            return float(arr)
        if len(arr) > 1:
            return float(arr[1])
        return float(arr[0])
    return float(value)


def validation_mean_auroc(results: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Mean validation AUROC across the three held-out regimes."""
    val_results = results[results["eval_type"] == SetNames.VAL]
    regime_scores = {}

    for regime in REGIMES:
        regime_df = val_results[val_results["eval_regime"] == regime]
        regime_scores[str(regime)] = roc_auc_score_binary(
            regime_df["label"], regime_df["prediction_prob"]
        )

    return float(pd.Series(regime_scores).mean()), regime_scores


def apply_params(model_args: Any, params: dict[str, Any]) -> Any:
    """Set dataclass attributes for a model config."""
    for name, value in params.items():
        setattr(model_args, name, value)
    return model_args


def candidate_grid() -> dict[type, list[dict[str, Any]]]:
    """Small validation grid for the report baselines."""
    logistic_candidates = [
        {
            "sklearn_pipeline_param_clf__C": c,
            "sklearn_pipeline_param_clf__class_weight": class_weight,
        }
        for c in [1.0, 2.0, 5.0]
        for class_weight in [None, "balanced"]
    ]

    knn_base = [
        {
            "sklearn_pipeline_param_clf__n_neighbors": n_neighbors,
            "sklearn_pipeline_param_clf__weights": weights,
            "sklearn_pipeline_param_clf__p": 2,
        }
        for n_neighbors in [3, 5]
        for weights in ["uniform", "distance"]
    ]
    knn_candidates = []
    for params in knn_base:
        knn_candidates.append(
            params
            | {
                "feature_set": "LOGISTIC_9",
                "item_level_features_modes": [ItemLevelFeaturesModes.LOGISTIC],
                "trial_level_feature_names": [],
            }
        )

    ensemble_base = [
        {
            "sklearn_pipeline_param_clf__lr_weight": lr_weight,
            "sklearn_pipeline_param_clf__knn_n_neighbors": n_neighbors,
            "sklearn_pipeline_param_clf__knn_weights": weights,
            "sklearn_pipeline_param_clf__knn_p": 2,
        }
        for lr_weight in [0.5]
        for n_neighbors in [3, 5]
        for weights in ["uniform", "distance"]
    ]
    ensemble_candidates = []
    for params in ensemble_base:
        ensemble_candidates.append(
            params
            | {
                "feature_set": "LOGISTIC_9",
                "item_level_features_modes": [ItemLevelFeaturesModes.LOGISTIC],
                "trial_level_feature_names": [],
            }
        )

    return {
        LogisticRegressionMLArgs: logistic_candidates,
        KNNMLArgs: knn_candidates,
        LRKNNEnsembleMLArgs: ensemble_candidates,
    }


def run_candidate(
    model_config: type,
    fold_index: int,
    params: dict[str, Any],
    overwrite_data: bool,
) -> pd.DataFrame:
    data_args = PoTeC_DE(fold_index=fold_index)
    trainer_args = TrainerML(
        seed=42,
        num_workers=0,
        overwrite_data=overwrite_data,
        wandb_project="PoTeC_DE_student_lr_knn_local",
        wandb_job_type=f"{model_config.__name__}_PoTeC_DE",
    )
    model_args = apply_params(model_config(), params)

    return process_single_run(
        data_args=data_args,
        trainer_args=trainer_args,
        model_args=model_args,
        fold_index=fold_index,
    )


def main() -> None:
    lf.seed_everything(42, workers=True, verbose=False)
    torch.set_float32_matmul_precision("high")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tuning_rows = []
    grids = candidate_grid()

    for fold_index in range(4):
        for model_config, candidates in grids.items():
            best_score = float("-inf")
            best_params: dict[str, Any] | None = None

            for candidate_index, params in enumerate(candidates):
                print(
                    f"Tuning {model_config.__name__}, fold {fold_index}, "
                    f"candidate {candidate_index + 1}/{len(candidates)}"
                )
                results = run_candidate(
                    model_config,
                    fold_index,
                    params,
                    overwrite_data=True,
                )
                mean_auroc, regime_scores = validation_mean_auroc(results)

                tuning_rows.append(
                    {
                        "model": model_config.__name__,
                        "fold_index": fold_index,
                        "candidate_index": candidate_index,
                        "validation_mean_auroc": mean_auroc,
                        "params": repr(params),
                        **{
                            f"val_auroc_{regime}": score
                            for regime, score in regime_scores.items()
                        },
                    }
                )

                if mean_auroc > best_score:
                    best_score = mean_auroc
                    best_params = params

            assert best_params is not None
            print(
                f"Selected {model_config.__name__}, fold {fold_index}: "
                f"validation mean AUROC={best_score:.3f}, params={best_params}"
            )
            # Rerun the selected candidate last so results/raw contains the
            # fold's selected model predictions.
            run_candidate(
                model_config,
                fold_index,
                best_params,
                overwrite_data=True,
            )

    tuning_log = pd.DataFrame(tuning_rows)
    tuning_log.to_csv(TUNING_LOG_PATH, index=False)
    print(f"\nSaved validation tuning log: {TUNING_LOG_PATH}")


if __name__ == "__main__":
    main()
