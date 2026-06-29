import unittest

import numpy as np

from src.configs.data import PoTeC_DE
from src.configs.models.ml.StackingEnsemble import (
    CORE_GAZE_FEATURES,
    StackingEnsembleHeterogeneousMLArgs,
    StackingEnsembleMLArgs,
    StackingEnsembleReadingSpeedMLArgs,
)
from src.configs.trainers import TrainerML
from src.models.models_ml import StackingEnsembleMLModel
from src.models.nested_stacking import NestedStackingClassifier
from src.run.single_run.run_stacking_cv import (
    select_balanced_accuracy_threshold,
)


class StackingEnsembleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.groups = np.repeat(np.arange(30), 8)
        self.features = self.rng.normal(size=(len(self.groups), 10))
        noise = self.rng.normal(scale=0.5, size=len(self.groups))
        self.labels = (
            self.features[:, 0] + 0.5 * self.features[:, 1] + noise > 0
        ).astype(int)
        model_args = StackingEnsembleMLArgs(
            random_forest_n_estimators=10,
            stacking_n_jobs=1,
            stacking_n_splits=3,
            tuning_n_splits=2,
            calibration_n_splits=2,
            base_logistic_c_grid=[1.0],
            base_logistic_class_weight_grid=[None],
            knn_n_neighbors_grid=[5],
            knn_weights_grid=['uniform'],
            svm_c_grid=[1.0],
            svm_gamma_grid=['scale'],
            random_forest_max_depth_grid=[4],
            random_forest_min_samples_leaf_grid=[2],
            meta_logistic_c_grid=[1.0],
            meta_logistic_penalty_grid=['l1'],
            meta_logistic_class_weight_grid=[None],
        )
        self.model = StackingEnsembleMLModel(
            model_args,
            TrainerML(),
            PoTeC_DE(),
        )

    def test_oof_folds_do_not_share_participants(self) -> None:
        splits = self.model._build_grouped_cv_splits(
            self.features,
            self.labels,
            self.groups.astype(str),
        )

        validation_indices = []
        for train_indices, validation_fold_indices in splits:
            train_groups = set(self.groups[train_indices])
            validation_groups = set(self.groups[validation_fold_indices])
            self.assertTrue(train_groups.isdisjoint(validation_groups))
            validation_indices.extend(validation_fold_indices.tolist())

        self.assertEqual(sorted(validation_indices), list(range(len(self.groups))))

    def test_meta_learner_receives_four_probability_features(self) -> None:
        classifier = NestedStackingClassifier(self.model.model_args)
        classifier.fit(
            self.features,
            self.labels,
            self.groups.astype(str),
        )

        probabilities = classifier.predict_proba(self.features)
        self.assertEqual(probabilities.shape, (len(self.features), 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertEqual(
            classifier.final_estimator_.named_steps['classifier'].coef_.shape,
            (1, 4),
        )
        self.assertFalse(np.isnan(classifier.oof_probabilities_).any())

    def test_feature_ablation_configs(self) -> None:
        core = StackingEnsembleMLArgs()
        with_reading_speed = StackingEnsembleReadingSpeedMLArgs()
        heterogeneous = StackingEnsembleHeterogeneousMLArgs()
        self.assertEqual(core.item_level_feature_names, CORE_GAZE_FEATURES)
        self.assertEqual(
            with_reading_speed.item_level_feature_names,
            [*CORE_GAZE_FEATURES, 'reading_speed'],
        )
        self.assertTrue(heterogeneous.include_reading_speed_base)
        self.assertTrue(heterogeneous.use_heterogeneous_feature_views)

    def test_heterogeneous_stack_has_five_meta_features(self) -> None:
        args = self.model.model_args
        args.include_reading_speed_base = True
        feature_indices = {
            'logistic_regression': np.array([0, 1, 2]),
            'knn': np.array([0, 1]),
            'svm_rbf': np.array([2, 3, 4]),
            'random_forest': np.arange(self.features.shape[1]),
            'reading_speed': np.array([5]),
        }
        classifier = NestedStackingClassifier(args, feature_indices)
        classifier.fit(self.features, self.labels, self.groups.astype(str))
        self.assertEqual(classifier.transform(self.features).shape, (240, 5))
        self.assertEqual(
            classifier.final_estimator_.named_steps['classifier'].coef_.shape,
            (1, 5),
        )

    def test_threshold_is_selected_from_balanced_accuracy(self) -> None:
        labels = np.array([0, 0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.4, 0.35, 0.45])
        threshold = select_balanced_accuracy_threshold(labels, probabilities)
        default_score = np.mean(
            [
                np.mean((probabilities[:3] >= 0.5) == labels[:3]),
                np.mean((probabilities[3:] >= 0.5) == labels[3:]),
            ]
        )
        tuned_predictions = probabilities >= threshold
        tuned_score = np.mean(
            [
                np.mean(tuned_predictions[:3] == labels[:3]),
                np.mean(tuned_predictions[3:] == labels[3:]),
            ]
        )
        self.assertGreater(tuned_score, default_score)


if __name__ == '__main__':
    unittest.main()
