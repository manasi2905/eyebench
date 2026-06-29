"""Run tuned PoTeC domain-expertise stacking experiments locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from src.configs.data import PoTeC_DE
from src.configs.models.ml.StackingEnsemble import (
    StackingEnsembleHeterogeneousMLArgs,
    StackingEnsembleMLArgs,
    StackingEnsembleReadingSpeedMLArgs,
)
from src.configs.trainers import TrainerML
from src.run.single_run.test_ml import process_single_run


MODEL_CONFIGS = {
    'core': StackingEnsembleMLArgs,
    'core_plus_reading_speed': StackingEnsembleReadingSpeedMLArgs,
    'heterogeneous': StackingEnsembleHeterogeneousMLArgs,
}
DEFAULT_PROBABILITY_COLUMNS = {
    'stacking_ensemble': 'positive_probability',
    'logistic_regression': 'base_logistic_regression_probability',
    'knn': 'base_knn_probability',
    'svm_rbf': 'base_svm_rbf_probability',
    'random_forest': 'base_random_forest_probability',
}


def probability_columns(predictions: pd.DataFrame) -> dict[str, str]:
    columns = DEFAULT_PROBABILITY_COLUMNS.copy()
    reading_speed_column = 'base_reading_speed_probability'
    if reading_speed_column in predictions:
        columns['reading_speed'] = reading_speed_column
    return columns


def parse_folds(raw_folds: str) -> list[int]:
    folds = [int(value.strip()) for value in raw_folds.split(',')]
    invalid_folds = [fold for fold in folds if fold not in range(4)]
    if invalid_folds:
        raise ValueError(f'PoTeC fold indices must be in [0, 3]: {invalid_folds}')
    return folds


def parse_feature_sets(raw_feature_sets: str) -> list[str]:
    feature_sets = [value.strip() for value in raw_feature_sets.split(',')]
    invalid_feature_sets = sorted(set(feature_sets) - set(MODEL_CONFIGS))
    if invalid_feature_sets:
        raise ValueError(f'Unknown feature sets: {invalid_feature_sets}')
    return feature_sets


def add_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions['positive_probability'] = predictions['prediction_prob'].apply(
        lambda values: float(np.asarray(values)[1])
    )
    predictions['predicted_label'] = (
        predictions['positive_probability'] >= 0.5
    ).astype(int)
    return predictions


def select_balanced_accuracy_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Select a threshold using validation labels only."""
    if len(np.unique(labels)) < 2:
        return 0.5
    unique_probabilities = np.unique(probabilities)
    candidates = np.concatenate(
        [
            [np.nextafter(unique_probabilities[0], -np.inf)],
            unique_probabilities,
            [np.nextafter(unique_probabilities[-1], np.inf)],
        ]
    )
    scores = np.array(
        [
            balanced_accuracy_score(labels, probabilities >= threshold)
            for threshold in candidates
        ]
    )
    best_candidates = candidates[np.isclose(scores, scores.max())]
    return float(min(best_candidates, key=lambda threshold: abs(threshold - 0.5)))


def add_validation_tuned_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions['decision_threshold'] = 0.5
    for (fold_index, eval_regime), group in predictions.groupby(
        ['fold_index', 'eval_regime'],
        sort=True,
    ):
        validation = group[group['eval_type'] == 'val']
        threshold = select_balanced_accuracy_threshold(
            validation['label'].astype(int).to_numpy(),
            validation['positive_probability'].to_numpy(),
        )
        mask = (
            (predictions['fold_index'] == fold_index)
            & (predictions['eval_regime'] == eval_regime)
        )
        predictions.loc[mask, 'decision_threshold'] = threshold
    predictions['predicted_label_tuned'] = (
        predictions['positive_probability'] >= predictions['decision_threshold']
    ).astype(int)
    return predictions


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ['fold_index', 'eval_type', 'eval_regime']
    for group_values, group in predictions.groupby(group_columns, sort=True):
        fold_index, eval_type, eval_regime = group_values
        labels = group['label'].astype(int).to_numpy()
        for model_name, probability_column in probability_columns(
            predictions
        ).items():
            probabilities = group[probability_column].to_numpy()
            default_predictions = (probabilities >= 0.5).astype(int)
            if model_name == 'stacking_ensemble':
                threshold = float(group['decision_threshold'].iloc[0])
            else:
                threshold = 0.5
            tuned_predictions = (probabilities >= threshold).astype(int)
            auroc = (
                roc_auc_score(labels, probabilities)
                if len(np.unique(labels)) == 2
                else np.nan
            )
            rows.append(
                {
                    'fold_index': fold_index,
                    'eval_type': eval_type,
                    'eval_regime': eval_regime,
                    'model': model_name,
                    'n_samples': len(group),
                    'positive_rate': labels.mean(),
                    'auroc': auroc,
                    'accuracy': accuracy_score(labels, default_predictions),
                    'balanced_accuracy': balanced_accuracy_score(
                        labels,
                        default_predictions,
                    ),
                    'decision_threshold': threshold,
                    'threshold_tuned_accuracy': accuracy_score(
                        labels,
                        tuned_predictions,
                    ),
                    'threshold_tuned_balanced_accuracy': balanced_accuracy_score(
                        labels,
                        tuned_predictions,
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(['eval_type', 'eval_regime', 'model'], sort=True)
        .agg(
            folds=('fold_index', 'nunique'),
            mean_auroc=('auroc', 'mean'),
            std_auroc=('auroc', 'std'),
            mean_accuracy=('accuracy', 'mean'),
            std_accuracy=('accuracy', 'std'),
            mean_balanced_accuracy=('balanced_accuracy', 'mean'),
            std_balanced_accuracy=('balanced_accuracy', 'std'),
            mean_threshold=('decision_threshold', 'mean'),
            mean_threshold_tuned_accuracy=('threshold_tuned_accuracy', 'mean'),
            mean_threshold_tuned_balanced_accuracy=(
                'threshold_tuned_balanced_accuracy',
                'mean',
            ),
        )
        .reset_index()
    )


def calculate_base_probability_correlations(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = list(probability_columns(predictions).values())[1:]
    rows = []
    group_columns = ['fold_index', 'eval_type', 'eval_regime']
    for group_values, group in predictions.groupby(group_columns, sort=True):
        fold_index, eval_type, eval_regime = group_values
        correlation = group[base_columns].corr()
        for first_index, first_model in enumerate(base_columns):
            for second_model in base_columns[first_index + 1 :]:
                rows.append(
                    {
                        'fold_index': fold_index,
                        'eval_type': eval_type,
                        'eval_regime': eval_regime,
                        'first_model': first_model.removeprefix('base_').removesuffix(
                            '_probability'
                        ),
                        'second_model': second_model.removeprefix(
                            'base_'
                        ).removesuffix('_probability'),
                        'correlation': correlation.loc[first_model, second_model],
                    }
                )
    return pd.DataFrame(rows)


def collect_oof_metrics(model_config, folds: list[int]) -> pd.DataFrame:
    rows = []
    raw_root = Path('results/raw')
    model_name = model_config.__name__
    directory_name = (
        f'+data=PoTeC_DE,+model={model_name},+trainer=TrainerML,'
        f'trainer.wandb_job_type={model_name}_PoTeC_DE'
    )
    for fold_index in folds:
        metrics_path = (
            raw_root
            / directory_name
            / f'fold_index={fold_index}'
            / 'stacking_oof_metrics.csv'
        )
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, 'fold_index', fold_index)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def run_feature_set(
    feature_set: str,
    folds: list[int],
    output_dir: Path,
) -> pd.DataFrame:
    model_config = MODEL_CONFIGS[feature_set]
    all_predictions = []
    for fold_index in folds:
        logger.info(
            f'Running tuned {feature_set} stacking ensemble on PoTeC_DE fold '
            f'{fold_index}'
        )
        predictions = process_single_run(
            data_args=PoTeC_DE(fold_index=fold_index),
            trainer_args=TrainerML(num_workers=0),
            model_args=model_config(),
            fold_index=fold_index,
        )
        all_predictions.append(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = add_prediction_columns(pd.concat(all_predictions, ignore_index=True))
    predictions = add_validation_tuned_thresholds(predictions)
    metrics = calculate_metrics(predictions)
    summary = summarize_metrics(metrics)
    correlations = calculate_base_probability_correlations(predictions)
    oof_metrics = collect_oof_metrics(model_config, folds)

    predictions.to_csv(output_dir / 'predictions.csv', index=False)
    metrics.to_csv(output_dir / 'metrics_by_fold.csv', index=False)
    summary.to_csv(output_dir / 'metrics_summary.csv', index=False)
    correlations.to_csv(output_dir / 'base_probability_correlations.csv', index=False)
    oof_metrics.to_csv(output_dir / 'oof_metrics_by_fold.csv', index=False)
    logger.info(f'Saved tuned stacking results to {output_dir}')
    print(f'\nFeature set: {feature_set}')
    print(summary[summary['eval_type'] == 'test'].to_string(index=False))
    return summary.assign(feature_set=feature_set)


def run_cross_validation(
    folds: list[int],
    feature_sets: list[str],
    output_dir: Path,
) -> None:
    summaries = [
        run_feature_set(feature_set, folds, output_dir / feature_set)
        for feature_set in feature_sets
    ]
    comparison = pd.concat(summaries, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / 'feature_set_comparison.csv', index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', default='0,1,2,3')
    parser.add_argument(
        '--feature-sets',
        default='core,core_plus_reading_speed',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/stacking_potec_de_tuned'),
    )
    args = parser.parse_args()
    run_cross_validation(
        parse_folds(args.folds),
        parse_feature_sets(args.feature_sets),
        args.output_dir,
    )


if __name__ == '__main__':
    main()
