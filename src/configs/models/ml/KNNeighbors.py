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
class KNNMLArgs(MLModelArgs):
    base_model_name: MLModelNames = MLModelNames.KNN

    sklearn_pipeline: tuple = (
        ('scaler', 'sklearn.preprocessing.StandardScaler'),
        ('clf', 'sklearn.neighbors.KNeighborsClassifier'),
    )

    # Ran the model for n_neighbors in [3, 5, 7], weights in ['uniform', 'distance'], and p in [1, 2].
    sklearn_pipeline_param_clf__n_neighbors: int = 3
    sklearn_pipeline_param_clf__weights: str = 'uniform'
    sklearn_pipeline_param_clf__metric: str = 'minkowski'
    sklearn_pipeline_param_clf__p: int = 2
    sklearn_pipeline_param_clf__n_jobs: int = -1

    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True

    batch_size: int = 1024
    use_fixation_report: bool = True

    # KNeighborsClassifier.fit() does not support sample_weight.
    use_class_weighted_loss: bool = False

    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE

    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.LOGISTIC],
    )