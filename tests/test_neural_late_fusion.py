import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.run.single_run.run_neural_late_fusion import (
    build_grouped_splits,
    calibrate_neural_probabilities,
    merge_prediction_sources,
    meta_learner_spec,
)


class NeuralLateFusionTest(unittest.TestCase):
    def test_prediction_sources_align_one_to_one(self) -> None:
        tabular = pd.DataFrame(
            {
                'fold_index': [0, 0],
                'eval_type': ['val', 'test'],
                'eval_regime': ['regime', 'regime'],
                'participant_id': ['p1', 'p2'],
                'unique_trial_id': ['t1', 't2'],
                'label': [0, 1],
            }
        )
        neural = tabular.copy()
        neural['neural_probability_raw'] = [0.2, 0.8]
        merged = merge_prediction_sources(tabular, neural)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged['neural_probability_raw'].tolist(), [0.2, 0.8])

    def test_neural_calibration_is_group_cross_fitted(self) -> None:
        rng = np.random.default_rng(42)
        groups = np.repeat(np.arange(30), 4).astype(str)
        labels = np.tile([0, 0, 1, 0], 30)
        raw_probabilities = np.clip(
            0.2 + 0.5 * labels + rng.normal(0, 0.1, len(labels)),
            0.01,
            0.99,
        )
        validation = pd.DataFrame(
            {
                'label': labels,
                'participant_id': groups,
                'neural_probability_raw': raw_probabilities,
            }
        )
        test = validation.iloc[:12].copy()
        splits = build_grouped_splits(labels, groups, requested_splits=3)
        validation_probabilities, test_probabilities, parameters = (
            calibrate_neural_probabilities(validation, test, splits)
        )
        self.assertEqual(validation_probabilities.shape, (120,))
        self.assertEqual(test_probabilities.shape, (12,))
        self.assertTrue(np.all((validation_probabilities > 0) & (validation_probabilities < 1)))
        self.assertIn('coefficient', parameters)

    def test_both_report_meta_learners_are_available(self) -> None:
        logistic, _ = meta_learner_spec('logistic_regression')
        mlp, mlp_grid = meta_learner_spec('mlp')
        self.assertIsInstance(logistic.named_steps['classifier'], LogisticRegression)
        self.assertIsInstance(mlp.named_steps['classifier'], MLPClassifier)
        self.assertEqual(
            max(size[0] for size in mlp_grid['classifier__hidden_layer_sizes']),
            4,
        )


if __name__ == '__main__':
    unittest.main()
