from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.configs.models.ml.StackingEnsemble import StackingEnsembleMLArgs


class NestedStackingClassifier(ClassifierMixin, BaseEstimator):
    """Two-layer classifier with nested tuning and group-safe cross-fitting."""

    default_base_model_names = (
        'logistic_regression',
        'knn',
        'svm_rbf',
        'random_forest',
    )

    def __init__(
        self,
        model_args: StackingEnsembleMLArgs,
        feature_indices: dict[str, np.ndarray] | None = None,
    ):
        self.model_args = model_args
        self.base_model_names = list(self.default_base_model_names)
        if model_args.include_reading_speed_base:
            self.base_model_names.append('reading_speed')
        self.feature_indices = feature_indices

    def _model_features(
        self,
        model_name: str,
        features: np.ndarray,
    ) -> np.ndarray:
        if self.feature_indices is None:
            return features
        return features[:, self.feature_indices[model_name]]

    def _build_grouped_splits(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        requested_splits: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        max_splits = min(requested_splits, len(np.unique(groups)))
        for n_splits in range(max_splits, 1, -1):
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.model_args.stacking_random_state,
            )
            splits = list(splitter.split(features, labels, groups=groups))
            if all(
                len(np.unique(labels[train_indices])) == 2
                and len(np.unique(labels[validation_indices])) == 2
                for train_indices, validation_indices in splits
            ):
                return splits
        raise ValueError(
            'Unable to create at least two participant-grouped folds with both '
            'classes in every training and validation partition.'
        )

    def _base_model_spec(
        self,
        model_name: str,
    ) -> tuple[BaseEstimator, dict[str, list[Any]]]:
        args = self.model_args
        random_state = args.stacking_random_state
        if model_name in {'logistic_regression', 'reading_speed'}:
            estimator = Pipeline(
                [
                    ('scaler', StandardScaler()),
                    (
                        'classifier',
                        LogisticRegression(
                            max_iter=args.base_logistic_max_iter,
                            random_state=random_state,
                            solver=args.base_logistic_solver,
                        ),
                    ),
                ]
            )
            grid = {
                'classifier__C': args.base_logistic_c_grid,
                'classifier__class_weight': args.base_logistic_class_weight_grid,
            }
        elif model_name == 'knn':
            estimator = Pipeline(
                [
                    ('scaler', StandardScaler()),
                    ('classifier', KNeighborsClassifier()),
                ]
            )
            grid = {
                'classifier__n_neighbors': args.knn_n_neighbors_grid,
                'classifier__weights': args.knn_weights_grid,
            }
        elif model_name == 'svm_rbf':
            estimator = Pipeline(
                [
                    ('scaler', StandardScaler()),
                    (
                        'classifier',
                        SVC(
                            kernel='rbf',
                            probability=False,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            grid = {
                'classifier__C': args.svm_c_grid,
                'classifier__gamma': args.svm_gamma_grid,
                'classifier__class_weight': [None, 'balanced'],
            }
        elif model_name == 'random_forest':
            estimator = RandomForestClassifier(
                n_estimators=args.random_forest_n_estimators,
                max_features='sqrt',
                n_jobs=1,
                random_state=random_state,
            )
            grid = {
                'max_depth': args.random_forest_max_depth_grid,
                'min_samples_leaf': args.random_forest_min_samples_leaf_grid,
                'class_weight': [None, 'balanced'],
            }
        else:
            raise ValueError(f'Unknown stacking base model: {model_name}')
        return estimator, grid

    def _positive_probability(
        self,
        estimator: BaseEstimator,
        features: np.ndarray,
    ) -> np.ndarray:
        probabilities = estimator.predict_proba(features)
        positive_index = int(np.flatnonzero(estimator.classes_ == 1)[0])
        return probabilities[:, positive_index]

    def _tune_and_fit_base_model(
        self,
        model_name: str,
        features: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> tuple[BaseEstimator, dict[str, Any]]:
        estimator, parameter_grid = self._base_model_spec(model_name)
        tuning_splits = self._build_grouped_splits(
            features,
            labels,
            groups,
            self.model_args.tuning_n_splits,
        )
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring='roc_auc',
            cv=tuning_splits,
            refit=True,
            n_jobs=self.model_args.stacking_n_jobs,
            error_score='raise',
        )
        search.fit(features, labels)
        fitted_estimator: BaseEstimator = search.best_estimator_

        if model_name in self.model_args.calibrated_base_model_names:
            calibration_splits = self._build_grouped_splits(
                features,
                labels,
                groups,
                self.model_args.calibration_n_splits,
            )
            fitted_estimator = CalibratedClassifierCV(
                estimator=clone(search.best_estimator_),
                method=self.model_args.calibration_method,
                cv=calibration_splits,
                n_jobs=self.model_args.stacking_n_jobs,
                ensemble=True,
            )
            fitted_estimator.fit(features, labels)

        return fitted_estimator, search.best_params_

    def _build_meta_estimator(self) -> Pipeline:
        return Pipeline(
            [
                ('scaler', StandardScaler()),
                (
                    'classifier',
                    LogisticRegression(
                        max_iter=self.model_args.meta_logistic_max_iter,
                        random_state=self.model_args.stacking_random_state,
                        solver=self.model_args.meta_logistic_solver,
                    ),
                ),
            ]
        )

    def _fit_meta_learner(
        self,
        oof_probabilities: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        stacking_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        parameter_grid = {
            'classifier__C': self.model_args.meta_logistic_c_grid,
            'classifier__penalty': self.model_args.meta_logistic_penalty_grid,
            'classifier__class_weight': (
                self.model_args.meta_logistic_class_weight_grid
            ),
        }
        search = GridSearchCV(
            estimator=self._build_meta_estimator(),
            param_grid=parameter_grid,
            scoring='roc_auc',
            cv=stacking_splits,
            refit=True,
            n_jobs=self.model_args.stacking_n_jobs,
            error_score='raise',
        )
        search.fit(oof_probabilities, labels)
        self.final_estimator_ = search.best_estimator_
        self.meta_best_params_ = search.best_params_

        cross_fitted_probabilities = cross_val_predict(
            clone(search.best_estimator_),
            oof_probabilities,
            labels,
            cv=stacking_splits,
            method='predict_proba',
            n_jobs=self.model_args.stacking_n_jobs,
        )
        positive_index = int(
            np.flatnonzero(self.final_estimator_.classes_ == 1)[0]
        )
        self.oof_ensemble_probabilities_ = cross_fitted_probabilities[
            :, positive_index
        ]

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> NestedStackingClassifier:
        self.classes_ = np.unique(labels)
        if not np.array_equal(self.classes_, np.array([0, 1])):
            raise ValueError('NestedStackingClassifier requires binary labels 0 and 1.')

        stacking_splits = self._build_grouped_splits(
            features,
            labels,
            groups,
            self.model_args.stacking_n_splits,
        )
        logger.info(
            f'Creating stacking probabilities with {len(stacking_splits)} '
            'participant-grouped outer folds.'
        )
        oof_probabilities = np.full(
            (len(features), len(self.base_model_names)),
            np.nan,
        )
        fold_assignments = np.full(len(features), -1, dtype=int)
        selected_hyperparameters = []

        for stacking_fold, (train_indices, validation_indices) in enumerate(
            stacking_splits
        ):
            fold_assignments[validation_indices] = stacking_fold
            logger.info(
                f'Tuning stacking outer fold {stacking_fold + 1}/'
                f'{len(stacking_splits)}.'
            )
            for model_index, model_name in enumerate(self.base_model_names):
                estimator, best_params = self._tune_and_fit_base_model(
                    model_name,
                    self._model_features(model_name, features[train_indices]),
                    labels[train_indices],
                    groups[train_indices],
                )
                oof_probabilities[validation_indices, model_index] = (
                    self._positive_probability(
                        estimator,
                        self._model_features(
                            model_name,
                            features[validation_indices],
                        ),
                    )
                )
                selected_hyperparameters.append(
                    {
                        'stacking_fold': stacking_fold,
                        'model': model_name,
                        **best_params,
                    }
                )

        if np.isnan(oof_probabilities).any() or np.any(fold_assignments < 0):
            raise RuntimeError('Stacking cross-fitting did not predict every sample.')

        self.oof_probabilities_ = oof_probabilities
        self.oof_labels_ = labels.copy()
        self.oof_groups_ = groups.copy()
        self.oof_fold_assignments_ = fold_assignments
        self.selected_hyperparameters_ = selected_hyperparameters
        self.base_oof_auroc_ = {
            model_name: roc_auc_score(labels, oof_probabilities[:, model_index])
            for model_index, model_name in enumerate(self.base_model_names)
        }
        self.base_oof_correlation_ = np.corrcoef(
            oof_probabilities,
            rowvar=False,
        )

        self._fit_meta_learner(
            oof_probabilities,
            labels,
            groups,
            stacking_splits,
        )
        self.ensemble_oof_auroc_ = roc_auc_score(
            labels,
            self.oof_ensemble_probabilities_,
        )

        self.estimators_ = []
        self.final_base_hyperparameters_ = {}
        logger.info('Tuning final base models on the complete outer-training set.')
        for model_name in self.base_model_names:
            estimator, best_params = self._tune_and_fit_base_model(
                model_name,
                self._model_features(model_name, features),
                labels,
                groups,
            )
            self.estimators_.append(estimator)
            self.final_base_hyperparameters_[model_name] = best_params

        logger.info(f'Base OOF AUROC: {self.base_oof_auroc_}')
        logger.info(f'Meta-learner parameters: {self.meta_best_params_}')
        logger.info(f'Cross-fitted ensemble OOF AUROC: {self.ensemble_oof_auroc_:.4f}')
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                self._positive_probability(
                    estimator,
                    self._model_features(model_name, features),
                )
                for model_name, estimator in zip(
                    self.base_model_names,
                    self.estimators_,
                    strict=True,
                )
            ]
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.final_estimator_.predict_proba(self.transform(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.final_estimator_.predict(self.transform(features))
