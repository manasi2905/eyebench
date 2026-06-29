from dataclasses import dataclass, field

from src.configs.constants import (
    BackboneNames,
    Fields,
    ItemLevelFeaturesModes,
    MLModelNames,
)
from src.configs.models.base_model import MLModelArgs
from src.configs.utils import register_model_config


CORE_GAZE_FEATURES = [
    'CURRENT_FIX_DURATION_mean',
    'num_of_fixations',
    'regression_rate',
    'skip_rate',
]

LOGISTIC_GAZE_FEATURES = [
    'CURRENT_FIX_DURATION_mean',
    'forward_saccade_length_mean',
    'regression_rate',
    'first_pass_skip_rate',
    'mean_FFD',
    'mean_GD',
    'mean_TFD',
    'mean_go_past_time',
    'reading_speed',
]

SVM_GAZE_FEATURES = [
    'NEXT_SAC_DURATION_mean',
    'NEXT_SAC_DURATION_max',
    'NEXT_SAC_AVG_VELOCITY_mean',
    'NEXT_SAC_AVG_VELOCITY_max',
    'NEXT_SAC_AMPLITUDE_mean',
    'NEXT_SAC_AMPLITUDE_max',
    'skip_rate',
    'num_of_fixations',
    'mean_TFD',
]


@register_model_config
@dataclass
class StackingEnsembleMLArgs(MLModelArgs):
    """Configuration for a tuned, leakage-safe two-layer ensemble."""

    base_model_name: MLModelNames = MLModelNames.STACKING_ENSEMBLE

    # BaseMLModel initializes this field, but the stacking model constructs its
    # four estimators and meta-learner directly.
    sklearn_pipeline: list = field(default_factory=list)

    batch_size: int = 1024
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    use_class_weighted_loss: bool = False
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(default_factory=list)
    item_level_feature_names: list[str] = field(
        default_factory=lambda: list(CORE_GAZE_FEATURES)
    )
    stacking_feature_set_name: str = 'core'

    # The outer cross-fitting loop creates honest probabilities for the
    # meta-learner. Every base-model search is nested inside one such split.
    stacking_n_splits: int = 4
    tuning_n_splits: int = 3
    calibration_n_splits: int = 3
    stacking_group_column: str = Fields.SUBJECT_ID
    stacking_n_jobs: int = -1
    stacking_random_state: int = 42
    calibration_method: str = 'sigmoid'
    calibrated_base_model_names: list[str] = field(
        default_factory=lambda: ['svm_rbf', 'random_forest']
    )
    include_reading_speed_base: bool = False
    use_heterogeneous_feature_views: bool = False

    base_logistic_c_grid: list[float] = field(
        default_factory=lambda: [0.1, 1.0, 10.0]
    )
    base_logistic_class_weight_grid: list[str | None] = field(
        default_factory=lambda: [None, 'balanced']
    )
    base_logistic_max_iter: int = 2000
    base_logistic_solver: str = 'liblinear'
    knn_n_neighbors_grid: list[int] = field(default_factory=lambda: [3, 7, 15])
    knn_weights_grid: list[str] = field(
        default_factory=lambda: ['uniform', 'distance']
    )
    svm_c_grid: list[float] = field(default_factory=lambda: [1.0, 10.0, 100.0])
    svm_gamma_grid: list[str | float] = field(
        default_factory=lambda: ['scale', 0.01, 0.1]
    )
    random_forest_n_estimators: int = 300
    random_forest_max_depth_grid: list[int | None] = field(
        default_factory=lambda: [4, 8, None]
    )
    random_forest_min_samples_leaf_grid: list[int] = field(
        default_factory=lambda: [1, 5, 10]
    )
    meta_logistic_c_grid: list[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0]
    )
    meta_logistic_penalty_grid: list[str] = field(
        default_factory=lambda: ['l1', 'l2']
    )
    meta_logistic_class_weight_grid: list[str | None] = field(
        default_factory=lambda: [None, 'balanced']
    )
    meta_logistic_max_iter: int = 2000
    meta_logistic_solver: str = 'liblinear'


@register_model_config
@dataclass
class StackingEnsembleReadingSpeedMLArgs(StackingEnsembleMLArgs):
    """Core proposal features plus the strong reading-speed baseline."""

    item_level_feature_names: list[str] = field(
        default_factory=lambda: [*CORE_GAZE_FEATURES, 'reading_speed']
    )
    stacking_feature_set_name: str = 'core_plus_reading_speed'


@register_model_config
@dataclass
class StackingEnsembleHeterogeneousMLArgs(StackingEnsembleMLArgs):
    """Five tabular bases using model-specific feature views."""

    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.RF,
            ItemLevelFeaturesModes.SVM,
            ItemLevelFeaturesModes.LOGISTIC,
            ItemLevelFeaturesModes.READING_SPEED,
        ]
    )
    item_level_feature_names: list[str] = field(default_factory=list)
    include_reading_speed_base: bool = True
    use_heterogeneous_feature_views: bool = True
    stacking_feature_set_name: str = 'heterogeneous'
