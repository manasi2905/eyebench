"""Fuse heterogeneous tabular bases with held-out neural probabilities."""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import OptimizeWarning

from src.run.single_run.run_stacking_cv import (
    run_feature_set,
    select_balanced_accuracy_threshold,
)


TRIAL_KEY_CANDIDATES = [
    'fold_index',
    'eval_type',
    'eval_regime',
    'unique_paragraph_id',
    'participant_id',
    'unique_trial_id',
    'question',
    'DE_RC',
    'RC',
    'DE',
]
TABULAR_INPUT_COLUMNS = [
    'base_logistic_regression_probability',
    'base_knn_probability',
    'base_svm_rbf_probability',
    'base_random_forest_probability',
    'base_reading_speed_probability',
]
FUSION_INPUT_COLUMNS = [*TABULAR_INPUT_COLUMNS, 'neural_probability']

warnings.filterwarnings(
    'ignore',
    message='Unknown solver options: iprint',
    category=OptimizeWarning,
)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def parse_probability(value: Any) -> float:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[1])
    return float(value)


def build_grouped_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    requested_splits: int = 4,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    placeholder_features = np.zeros((len(labels), 1))
    max_splits = min(requested_splits, len(np.unique(groups)))
    for n_splits in range(max_splits, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(placeholder_features, labels, groups))
        if all(
            len(np.unique(labels[train_indices])) == 2
            and len(np.unique(labels[validation_indices])) == 2
            for train_indices, validation_indices in splits
        ):
            return splits
    raise ValueError('Could not create group-safe fusion folds with both classes.')


def load_neural_predictions(
    neural_model: str,
    raw_results_dir: Path,
) -> pd.DataFrame:
    roots = list(
        raw_results_dir.glob(f'+data=PoTeC_DE,+model={neural_model},*')
    )
    if len(roots) != 1:
        raise FileNotFoundError(
            f'Expected one PoTeC_DE result directory for {neural_model}, found '
            f'{len(roots)}.'
        )
    prediction_files = sorted(
        roots[0].glob('fold_index=*/trial_level_test_results.csv')
    )
    if len(prediction_files) != 4:
        raise FileNotFoundError(
            f'Expected four {neural_model} fold files, found {len(prediction_files)}.'
        )
    predictions = pd.concat(
        [pd.read_csv(path) for path in prediction_files],
        ignore_index=True,
    )
    predictions['neural_probability_raw'] = predictions['prediction_prob'].map(
        parse_probability
    )
    return predictions


def merge_prediction_sources(
    tabular_predictions: pd.DataFrame,
    neural_predictions: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        column
        for column in TRIAL_KEY_CANDIDATES
        if column in tabular_predictions and column in neural_predictions
    ]
    if tabular_predictions.duplicated(keys).any() or neural_predictions.duplicated(
        keys
    ).any():
        raise ValueError('Fusion keys do not uniquely identify predictions.')
    neural_columns = keys + ['label', 'neural_probability_raw']
    merged = tabular_predictions.merge(
        neural_predictions[neural_columns],
        on=keys,
        how='inner',
        validate='one_to_one',
        suffixes=('', '_neural'),
    )
    if len(merged) != len(tabular_predictions):
        raise ValueError(
            f'Only {len(merged)} of {len(tabular_predictions)} tabular predictions '
            'matched the neural predictions.'
        )
    if not np.array_equal(merged['label'], merged['label_neural']):
        raise ValueError('Tabular and neural prediction labels disagree.')
    return merged.drop(columns=['label_neural'])


def probability_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def calibrate_neural_probabilities(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    validation_logits = probability_logit(
        validation['neural_probability_raw'].to_numpy()
    )
    test_logits = probability_logit(test['neural_probability_raw'].to_numpy())
    labels = validation['label'].astype(int).to_numpy()
    calibrator = LogisticRegression(C=1.0, max_iter=2000, solver='liblinear')
    validation_probabilities = cross_val_predict(
        calibrator,
        validation_logits,
        labels,
        cv=splits,
        method='predict_proba',
        n_jobs=-1,
    )[:, 1]
    calibrator.fit(validation_logits, labels)
    test_probabilities = calibrator.predict_proba(test_logits)[:, 1]
    calibration_parameters = {
        'coefficient': float(calibrator.coef_[0, 0]),
        'intercept': float(calibrator.intercept_[0]),
    }
    return validation_probabilities, test_probabilities, calibration_parameters


def meta_learner_spec(
    meta_learner: str,
) -> tuple[Pipeline, dict[str, list[Any]]]:
    if meta_learner == 'logistic_regression':
        estimator = Pipeline(
            [
                ('scaler', StandardScaler()),
                (
                    'classifier',
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                        solver='liblinear',
                    ),
                ),
            ]
        )
        parameter_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__class_weight': [None, 'balanced'],
        }
    elif meta_learner == 'mlp':
        estimator = Pipeline(
            [
                ('scaler', StandardScaler()),
                (
                    'classifier',
                    MLPClassifier(
                        max_iter=2000,
                        random_state=42,
                        solver='lbfgs',
                    ),
                ),
            ]
        )
        parameter_grid = {
            'classifier__hidden_layer_sizes': [(2,), (4,)],
            'classifier__activation': ['tanh', 'relu'],
            'classifier__alpha': [0.01, 0.1, 1.0, 10.0],
        }
    else:
        raise ValueError(f'Unknown meta-learner: {meta_learner}')
    return estimator, parameter_grid


def fit_meta_learner(
    meta_learner: str,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any], BaseEstimator]:
    estimator, parameter_grid = meta_learner_spec(meta_learner)
    validation_features = validation[FUSION_INPUT_COLUMNS].to_numpy()
    test_features = test[FUSION_INPUT_COLUMNS].to_numpy()
    labels = validation['label'].astype(int).to_numpy()
    search = GridSearchCV(
        estimator,
        parameter_grid,
        scoring='roc_auc',
        cv=splits,
        refit=True,
        n_jobs=1,
        error_score='raise',
    )
    search.fit(validation_features, labels)
    validation_probabilities = cross_val_predict(
        clone(search.best_estimator_),
        validation_features,
        labels,
        cv=splits,
        method='predict_proba',
        n_jobs=1,
    )[:, 1]
    test_probabilities = search.best_estimator_.predict_proba(test_features)[:, 1]
    threshold = select_balanced_accuracy_threshold(
        labels,
        validation_probabilities,
    )
    return (
        validation_probabilities,
        test_probabilities,
        threshold,
        search.best_params_,
        search.best_estimator_,
    )


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_values, group in predictions.groupby(
        ['fold_index', 'eval_regime', 'meta_learner'],
        sort=True,
    ):
        fold_index, eval_regime, meta_learner = group_values
        labels = group['label'].astype(int).to_numpy()
        probabilities = group['fusion_probability'].to_numpy()
        threshold = float(group['decision_threshold'].iloc[0])
        default_predictions = probabilities >= 0.5
        tuned_predictions = probabilities >= threshold
        rows.append(
            {
                'fold_index': fold_index,
                'eval_regime': eval_regime,
                'meta_learner': meta_learner,
                'n_samples': len(group),
                'positive_rate': labels.mean(),
                'auroc': roc_auc_score(labels, probabilities),
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
        metrics.groupby(['eval_regime', 'meta_learner'], sort=True)
        .agg(
            folds=('fold_index', 'nunique'),
            mean_auroc=('auroc', 'mean'),
            std_auroc=('auroc', 'std'),
            mean_accuracy=('accuracy', 'mean'),
            mean_balanced_accuracy=('balanced_accuracy', 'mean'),
            mean_threshold=('decision_threshold', 'mean'),
            mean_threshold_tuned_accuracy=('threshold_tuned_accuracy', 'mean'),
            mean_threshold_tuned_balanced_accuracy=(
                'threshold_tuned_balanced_accuracy',
                'mean',
            ),
        )
        .reset_index()
    )


def calculate_input_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    input_columns = {
        **{
            column.removeprefix('base_').removesuffix('_probability'): column
            for column in TABULAR_INPUT_COLUMNS
        },
        'neural_raw': 'neural_probability_raw',
        'neural_calibrated': 'neural_probability',
    }
    rows = []
    for (fold_index, eval_regime), group in predictions.groupby(
        ['fold_index', 'eval_regime'],
        sort=True,
    ):
        labels = group['label'].astype(int).to_numpy()
        for model_name, column in input_columns.items():
            rows.append(
                {
                    'fold_index': fold_index,
                    'eval_regime': eval_regime,
                    'model': model_name,
                    'auroc': roc_auc_score(labels, group[column]),
                }
            )
    return pd.DataFrame(rows)


def serialize_meta_estimator(estimator: Pipeline) -> dict[str, Any]:
    classifier = estimator.named_steps['classifier']
    if isinstance(classifier, LogisticRegression):
        return {
            'coefficients': dict(
                zip(
                    FUSION_INPUT_COLUMNS,
                    (float(value) for value in classifier.coef_[0]),
                    strict=True,
                )
            ),
            'intercept': float(classifier.intercept_[0]),
        }
    return {
        'hidden_layer_sizes': classifier.hidden_layer_sizes,
        'loss': float(classifier.loss_),
        'n_iter': int(classifier.n_iter_),
    }


def run_late_fusion(
    tabular_predictions_path: Path,
    neural_model: str,
    meta_learners: list[str],
    output_dir: Path,
) -> None:
    tabular_predictions = pd.read_csv(tabular_predictions_path)
    neural_predictions = load_neural_predictions(neural_model, Path('results/raw'))
    merged = merge_prediction_sources(tabular_predictions, neural_predictions)

    test_prediction_frames = []
    validation_prediction_frames = []
    hyperparameters = []
    calibrated_test_frames = []
    for fold_index, fold in merged.groupby('fold_index', sort=True):
        validation = fold[fold['eval_type'] == 'val'].copy().reset_index(drop=True)
        test = fold[fold['eval_type'] == 'test'].copy().reset_index(drop=True)
        labels = validation['label'].astype(int).to_numpy()
        groups = validation['participant_id'].astype(str).to_numpy()
        splits = build_grouped_splits(labels, groups)

        (
            validation['neural_probability'],
            test['neural_probability'],
            calibration_parameters,
        ) = calibrate_neural_probabilities(validation, test, splits)
        calibrated_test_frames.append(test)

        for meta_learner in meta_learners:
            logger.info(
                f'Fitting {meta_learner} neural late fusion for fold {fold_index}.'
            )
            (
                validation_probabilities,
                test_probabilities,
                threshold,
                best_params,
                estimator,
            ) = fit_meta_learner(meta_learner, validation, test, splits)

            validation_output = validation.copy()
            validation_output['neural_model'] = neural_model
            validation_output['meta_learner'] = meta_learner
            validation_output['fusion_probability'] = validation_probabilities
            validation_output['decision_threshold'] = threshold
            validation_prediction_frames.append(validation_output)

            test_output = test.copy()
            test_output['neural_model'] = neural_model
            test_output['meta_learner'] = meta_learner
            test_output['fusion_probability'] = test_probabilities
            test_output['decision_threshold'] = threshold
            test_prediction_frames.append(test_output)

            hyperparameters.append(
                {
                    'fold_index': int(fold_index),
                    'meta_learner': meta_learner,
                    'neural_model': neural_model,
                    'neural_calibration': calibration_parameters,
                    'best_params': best_params,
                    'fitted_estimator': serialize_meta_estimator(estimator),
                }
            )

    test_predictions = pd.concat(test_prediction_frames, ignore_index=True)
    validation_predictions = pd.concat(
        validation_prediction_frames,
        ignore_index=True,
    )
    calibrated_test = pd.concat(calibrated_test_frames, ignore_index=True)
    metrics = calculate_metrics(test_predictions)
    summary = summarize_metrics(metrics)
    input_metrics = calculate_input_metrics(calibrated_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(output_dir / 'test_predictions.csv', index=False)
    validation_predictions.to_csv(
        output_dir / 'meta_training_oof_predictions.csv',
        index=False,
    )
    metrics.to_csv(output_dir / 'metrics_by_fold.csv', index=False)
    summary.to_csv(output_dir / 'metrics_summary.csv', index=False)
    input_metrics.to_csv(output_dir / 'input_model_metrics.csv', index=False)
    with (output_dir / 'selected_hyperparameters.json').open(
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(hyperparameters, file, indent=2, sort_keys=True)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--neural-model', default='RoberteyeFixation')
    parser.add_argument(
        '--meta-learners',
        default='logistic_regression,mlp',
    )
    parser.add_argument(
        '--tabular-results',
        type=Path,
        default=Path(
            'results/stacking_potec_de_tuned/heterogeneous/predictions.csv'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
    )
    parser.add_argument('--rebuild-tabular', action='store_true')
    args = parser.parse_args()

    if args.rebuild_tabular or not args.tabular_results.exists():
        run_feature_set(
            'heterogeneous',
            [0, 1, 2, 3],
            args.tabular_results.parent,
        )
    meta_learners = [value.strip() for value in args.meta_learners.split(',')]
    output_dir = args.output_dir
    if output_dir is None:
        suffix = '' if args.neural_model == 'RoberteyeFixation' else f'_{args.neural_model}'
        output_dir = Path(f'results/stacking_potec_de_neural_fusion{suffix}')
    run_late_fusion(
        args.tabular_results,
        args.neural_model,
        meta_learners,
        output_dir,
    )


if __name__ == '__main__':
    main()
