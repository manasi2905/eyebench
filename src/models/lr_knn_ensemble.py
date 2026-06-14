import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


class LRKNNEnsembleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        lr_weight: float = 0.5,
        knn_n_neighbors: int = 3,
        knn_weights: str = 'uniform',
        knn_p: int = 2,
    ):
        self.lr_weight = lr_weight
        self.knn_n_neighbors = knn_n_neighbors
        self.knn_weights = knn_weights
        self.knn_p = knn_p

    def fit(self, X, y):
        self.lr_model_ = Pipeline(
            [
                ('scaler', StandardScaler()),
                (
                    'clf',
                    LogisticRegression(
                        C=2.0,
                        penalty='l2',
                        solver='lbfgs',
                        max_iter=1000,
                        random_state=1,
                    ),
                ),
            ]
        )

        self.knn_model_ = Pipeline(
            [
                ('scaler', StandardScaler()),
                (
                    'clf',
                    KNeighborsClassifier(
                        n_neighbors=self.knn_n_neighbors,
                        weights=self.knn_weights,
                        metric='minkowski',
                        p=self.knn_p,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        lr_sample_weights = compute_sample_weight('balanced', y=y)

        self.lr_model_.fit(
            X,
            y,
            clf__sample_weight=lr_sample_weights,
        )
        self.knn_model_.fit(X, y)

        self.classes_ = self.lr_model_.classes_
        return self

    def predict_proba(self, X):
        lr_probs = self.lr_model_.predict_proba(X)
        knn_probs = self.knn_model_.predict_proba(X)

        return (
            self.lr_weight * lr_probs
            + (1.0 - self.lr_weight) * knn_probs
        )

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]