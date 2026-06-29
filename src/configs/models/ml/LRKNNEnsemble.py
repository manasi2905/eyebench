from dataclasses import dataclass, field

from src.configs.constants import (
    BackboneNames,
    ItemLevelFeaturesModes,
    MLModelNames,
)
from src.configs.models.base_model import MLModelArgs
from src.configs.utils import register_model_config


@register_model_config
@dataclass
class LRKNNEnsembleMLArgs(MLModelArgs):
    base_model_name: MLModelNames = MLModelNames.LR_KNN_ENSEMBLE

    sklearn_pipeline: tuple = (
        ('clf', 'src.models.lr_knn_ensemble.LRKNNEnsembleClassifier'),
    )

    sklearn_pipeline_param_clf__lr_weight: float = 0.5
    sklearn_pipeline_param_clf__knn_n_neighbors: int = 3
    sklearn_pipeline_param_clf__knn_weights: str = 'uniform'
    sklearn_pipeline_param_clf__knn_p: int = 2

    # Ensemble handles LR weighting internally.
    use_class_weighted_loss: bool = False

    batch_size: int = 1024
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE

    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.LOGISTIC],
    )