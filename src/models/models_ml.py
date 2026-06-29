import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger
from torch.utils.data import DataLoader

from src.configs.data import DataArgs
from src.configs.models.ml.DummyClassifier import (
    DummyClassifierMLArgs,
    DummyRegressorMLArgs,
)
from src.configs.models.ml.LogisticRegression import (
    LinearRegressionArgs,
    LogisticRegressionMLArgs,
)
from src.configs.models.ml.RandomForest import (
    RandomForestMLArgs,
    RandomForestRegressorMLArgs,
)
from src.configs.models.ml.SVM import (
    SupportVectorMachineMLArgs,
    SupportVectorRegressorMLArgs,
)
from src.configs.models.ml.StackingEnsemble import (
    CORE_GAZE_FEATURES,
    LOGISTIC_GAZE_FEATURES,
    SVM_GAZE_FEATURES,
    StackingEnsembleMLArgs,
)
from src.configs.models.ml.XGBoost import XGBoostMLArgs, XGBoostRegressorMLArgs
from src.configs.trainers import TrainerML
from src.models.base_model import BaseMLModel, register_model
from src.models.nested_stacking import NestedStackingClassifier


@register_model
class LogisticRegressionMLModel(BaseMLModel):
    """
    Logistic Regression classifier.
    """

    def __init__(
        self,
        model_args: LogisticRegressionMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class StackingEnsembleMLModel(BaseMLModel):
    """Nested two-layer ensemble trained on participant-grouped probabilities."""

    def __init__(
        self,
        model_args: StackingEnsembleMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args,
            trainer_args=trainer_args,
            data_args=data_args,
        )
        self.model_args = model_args
        self.data_args = data_args
        base_model_names = list(NestedStackingClassifier.default_base_model_names)
        if model_args.include_reading_speed_base:
            base_model_names.append('reading_speed')
        self.meta_feature_names = [f'{name}_probability' for name in base_model_names]
        self.meta_coefficients_: dict[str, float] = {}

    def _build_feature_indices(self) -> dict[str, np.ndarray] | None:
        if not self.model_args.use_heterogeneous_feature_views:
            return None
        if self.pca_explained_variance_ratio_threshold < 1.0:
            raise ValueError('PCA is incompatible with heterogeneous feature views.')
        if self.trial_level_feature_names is None:
            raise ValueError('Trial-level feature names are required for feature views.')

        feature_key_frames = [
            pd.read_csv(
                self.data_args.processed_data_path
                / 'fixation_trial_level_feature_keys.csv'
            ),
            pd.read_csv(
                self.data_args.processed_data_path
                / 'ia_trial_level_feature_keys.csv'
            ),
        ]
        feature_keys = pd.concat(feature_key_frames, ignore_index=True)
        random_forest_features = (
            feature_keys.loc[feature_keys['feature_type'] == 'RF', 'feature_name']
            .drop_duplicates()
            .tolist()
        )
        feature_views = {
            'logistic_regression': LOGISTIC_GAZE_FEATURES,
            'knn': CORE_GAZE_FEATURES,
            'svm_rbf': SVM_GAZE_FEATURES,
            'random_forest': random_forest_features,
            'reading_speed': ['reading_speed'],
        }
        feature_positions = {
            name: index for index, name in enumerate(self.trial_level_feature_names)
        }
        feature_indices = {}
        for model_name, requested_features in feature_views.items():
            missing_features = sorted(set(requested_features) - set(feature_positions))
            if missing_features:
                raise ValueError(
                    f'{model_name} feature view is missing: {missing_features}'
                )
            feature_indices[model_name] = np.array(
                [feature_positions[name] for name in requested_features],
                dtype=int,
            )
            logger.info(
                f'{model_name} receives {len(feature_indices[model_name])} features.'
            )
        return feature_indices

    def _get_group_values(
        self,
        trial_group_keys: np.ndarray,
        trial_key_columns: list[str],
    ) -> np.ndarray:
        group_column = self.model_args.stacking_group_column
        try:
            group_index = trial_key_columns.index(group_column)
        except ValueError as exc:
            raise ValueError(
                f'Stacking group column {group_column!r} is not present in '
                f'trial keys: {trial_key_columns}'
            ) from exc
        return trial_group_keys[:, group_index].astype(str)

    def _build_grouped_cv_splits(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return NestedStackingClassifier(self.model_args)._build_grouped_splits(
            features,
            labels,
            groups,
            self.model_args.stacking_n_splits,
        )

    def fit(self, dm) -> None:
        train_batches = self.shared_fit(dm)
        features, labels, trial_group_keys, trial_key_columns = (
            self._prepare_features_and_labels(train_batches, training=True)
        )
        labels = self.label_encoder.fit_transform(labels)

        if self.pca_explained_variance_ratio_threshold < 1.0:
            features = self._apply_pca(features)

        groups = self._get_group_values(trial_group_keys, trial_key_columns)
        self.classifier = NestedStackingClassifier(
            self.model_args,
            feature_indices=self._build_feature_indices(),
        )
        self.classifier.fit(features, labels, groups)

        coefficients = self.classifier.final_estimator_.named_steps[
            'classifier'
        ].coef_[0]
        self.meta_coefficients_ = dict(
            zip(
                self.meta_feature_names,
                (float(coefficient) for coefficient in coefficients),
                strict=True,
            )
        )
        logger.info(f'Stacking meta-learner coefficients: {self.meta_coefficients_}')
        if wandb.run is not None:
            wandb.log(
                {
                    f'stacking/meta_coefficient/{name}': value
                    for name, value in self.meta_coefficients_.items()
                }
            )

    def predict_base_probabilities(self, dataset) -> np.ndarray:
        """Return one positive-class probability column per fitted base model."""
        dev_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        dev_batches = self.unpack_data(dev_dataloader)
        features = np.concatenate(
            [
                self._features_builder(batch).to('cpu').numpy()
                for batch in dev_batches
            ],
            axis=0,
        )
        if self.pca_explained_variance_ratio_threshold < 1.0:
            features = self._apply_pca(features)
        return self.classifier.transform(features)


@register_model
class DummyClassifierMLModel(BaseMLModel):
    """
    Dummy classifier that uses null features and a dummy classifier from sklearn.
    """

    def __init__(
        self,
        model_args: DummyClassifierMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )

    def fit(self, dm) -> None:
        # Dummy classifier: generate zero features just for demonstration
        train_batches = self.shared_fit(dm)
        features_list = []
        y_true_list = []
        for train_batch in train_batches:
            features = torch.zeros((train_batch.labels.shape[0], 1)).to('cpu')
            features_list.append(features)
            y_true_list.append(train_batch.labels)

        features = torch.cat(features_list, dim=0).numpy()
        y_true = torch.cat(y_true_list, dim=0).numpy()
        self.classifier.fit(features, y_true)

    def model_specific_predict(
        self, dev_batches: list
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        preds_list: list[torch.Tensor] = []
        probs_list: list[torch.Tensor] = []
        for dev_batch in dev_batches:
            features = torch.zeros((dev_batch.labels.shape[0], 1)).to('cpu').numpy()
            preds, probs = self._predict_with_fallback(features)
            preds_list.append(preds)
            probs_list.append(probs)
        return preds_list, probs_list


@register_model
class SupportVectorMachineMLModel(BaseMLModel):
    """
    Support Vector Machine classifier.
    """

    def __init__(
        self,
        model_args: SupportVectorMachineMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class XGBoostMLModel(BaseMLModel):
    """
    XGBoost classifier.
    """

    def __init__(
        self, model_args: XGBoostMLArgs, trainer_args: TrainerML, data_args=DataArgs
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class RandomForestMLModel(BaseMLModel):
    """
    RandomForest classifier.
    """

    def __init__(
        self,
        model_args: RandomForestMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class LinearRegressionRegressorMLModel(BaseMLModel):
    """
    Logistic Regression classifier.
    """

    def __init__(
        self,
        model_args: LinearRegressionArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class DummyRegressorMLModel(BaseMLModel):
    """
    Dummy regressor that uses null features and a dummy regressor from sklearn.
    """

    def __init__(
        self,
        model_args: DummyRegressorMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )

    def fit(self, dm) -> None:
        # Dummy classifier: generate zero features just for demonstration
        train_batches = self.shared_fit(dm)
        features_list = []
        y_true_list = []
        for train_batch in train_batches:
            features = torch.zeros((train_batch.labels.shape[0], 1)).to('cpu')
            features_list.append(features)
            y_true_list.append(train_batch.labels)

        features = torch.cat(features_list, dim=0).numpy()
        y_true = torch.cat(y_true_list, dim=0).numpy()
        self.classifier.fit(features, y_true)

    def model_specific_predict(
        self, dev_batches: list
    ) -> tuple[list[torch.Tensor], list[None]]:
        preds_list: list[torch.Tensor] = []
        probs_list: list[None] = []
        for dev_batch in dev_batches:
            features = torch.zeros((dev_batch.labels.shape[0], 1)).to('cpu').numpy()
            preds = torch.tensor(self.classifier.predict(features))
            probs = None
            preds_list.append(preds)
            probs_list.append(probs)
        return preds_list, probs_list


@register_model
class SupportVectorRegressorMLModel(BaseMLModel):
    """
    Support Vector Machine regressor.
    """

    def __init__(
        self,
        model_args: SupportVectorRegressorMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class XGBoostRegressorMLModel(BaseMLModel):
    """
    XGBoost regressor.
    """

    def __init__(
        self,
        model_args: XGBoostRegressorMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class RandomForestRegressorMLModel(BaseMLModel):
    """
    RandomForest Regressor.
    """

    def __init__(
        self,
        model_args: RandomForestRegressorMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )
