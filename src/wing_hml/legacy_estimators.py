"""Self-contained local estimators adapted from the existing Wingedaircraft experiment.

MARS and XGBoost here are explicitly simplified implementations, not upstream
packages. The local multilinear model is weighted regression, not exact grid
interpolation. No imports or data from outside this project are required.
"""
from __future__ import annotations
import math
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

def solve_ridge(design: np.ndarray, target: np.ndarray, alpha: float, penalize_intercept: bool = False) -> np.ndarray:
    """Stable closed-form ridge solution with an optional free intercept."""
    gram = design.T @ design
    penalty = np.eye(design.shape[1]) * alpha
    if not penalize_intercept:
        penalty[0, 0] = 0.0
    return np.linalg.pinv(gram + penalty) @ design.T @ target


class RBFBasisRegressor(RegressorMixin, BaseEstimator):
    """Gaussian radial-basis network with deterministic centre selection."""

    def __init__(self, n_centers: int | None = None, gamma: float | None = None, alpha: float = 1e-6):
        self.n_centers = n_centers
        self.gamma = gamma
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y, dtype=float)
        if self.n_centers is None or self.n_centers >= len(X):
            # RBF surrogate specification: every training sample is a centre.
            self.centers_ = X.copy()
        else:
            count = min(self.n_centers, len(X))
            centre_indices = [int(np.argmin(np.sum(X * X, axis=1)))]
            minimum_distance = np.full(len(X), np.inf)
            for _ in range(1, count):
                latest = X[centre_indices[-1]]
                minimum_distance = np.minimum(minimum_distance, np.sum((X - latest) ** 2, axis=1))
                minimum_distance[centre_indices] = -1.0
                centre_indices.append(int(np.argmax(minimum_distance)))
            self.centers_ = X[centre_indices]

        pairwise = np.sum((self.centers_[:, None, :] - self.centers_[None, :, :]) ** 2, axis=2)
        nonzero = pairwise[pairwise > 1e-12]
        self.gamma_ = float(self.gamma) if self.gamma is not None else 1.0 / max(float(np.median(nonzero)), 1e-12)
        basis = np.exp(-self.gamma_ * np.sum((X[:, None, :] - self.centers_[None, :, :]) ** 2, axis=2))
        design = np.column_stack([np.ones(len(X)), basis])
        self.coef_ = solve_ridge(design, y, self.alpha)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("centers_", "coef_", "gamma_"))
        X = check_array(X, dtype=float)
        basis = np.exp(-self.gamma_ * np.sum((X[:, None, :] - self.centers_[None, :, :]) ** 2, axis=2))
        return np.column_stack([np.ones(len(X)), basis]) @ self.coef_


class TensorProductSplineTransformer(TransformerMixin, BaseEstimator):
    """Tensor product of one-dimensional B-spline bases."""

    def __init__(self, n_knots: int = 3, degree: int = 2):
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        X = check_array(X, dtype=float)
        self.splines_ = []
        for column in range(X.shape[1]):
            spline = SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                knots="quantile",
                include_bias=True,
            )
            spline.fit(X[:, [column]])
            self.splines_.append(spline)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "splines_")
        X = check_array(X, dtype=float)
        bases = [spline.transform(X[:, [index]]) for index, spline in enumerate(self.splines_)]
        result = bases[0]
        for basis in bases[1:]:
            result = np.einsum("ij,ik->ijk", result, basis).reshape(len(X), -1)
        return result


def _hinge(values: np.ndarray, knot: float, direction: int) -> np.ndarray:
    return np.maximum(0.0, direction * (values - knot))


class MARSRegressor(RegressorMixin, BaseEstimator):
    """Compact MARS implementation using hinge bases and forward selection.

    Candidate terms contain first-order hinges and products of hinges from two
    different input variables.  Forward selection minimises training RSS; a
    small ridge penalty stabilises the final fit.
    """

    def __init__(self, max_terms: int = 14, max_degree: int = 2, alpha: float = 1e-5, min_improvement: float = 1e-5):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.alpha = alpha
        self.min_improvement = min_improvement

    @staticmethod
    def _term_values(X: np.ndarray, term: tuple) -> np.ndarray:
        if term[0] == "hinge":
            _, column, knot, direction = term
            return _hinge(X[:, column], knot, direction)
        _, left, right = term
        return MARSRegressor._term_values(X, left) * MARSRegressor._term_values(X, right)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y, dtype=float)
        hinges: list[tuple] = []
        for column in range(X.shape[1]):
            unique = np.unique(X[:, column])
            knots = unique[1:-1] if len(unique) <= 8 else np.quantile(unique, [0.2, 0.4, 0.6, 0.8])
            for knot in knots:
                hinges.extend(
                    [
                        ("hinge", column, float(knot), 1),
                        ("hinge", column, float(knot), -1),
                    ]
                )
        candidates = list(hinges)
        if self.max_degree >= 2:
            for left_index, left in enumerate(hinges):
                for right in hinges[left_index + 1 :]:
                    if left[1] != right[1]:
                        candidates.append(("interaction", left, right))

        candidate_matrix = np.column_stack([self._term_values(X, term) for term in candidates]) if candidates else np.empty((len(X), 0))
        usable = np.std(candidate_matrix, axis=0) > 1e-12
        candidate_matrix = candidate_matrix[:, usable]
        candidates = [term for term, keep in zip(candidates, usable) if keep]

        selected: list[int] = []
        current_rss = float(np.sum((y - np.mean(y)) ** 2))
        initial_rss = max(current_rss, 1e-12)
        for _ in range(min(self.max_terms, len(candidates))):
            best_index = None
            best_rss = current_rss
            for candidate_index in range(len(candidates)):
                if candidate_index in selected:
                    continue
                columns = selected + [candidate_index]
                design = np.column_stack([np.ones(len(X)), candidate_matrix[:, columns]])
                coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
                rss = float(np.sum((y - design @ coefficients) ** 2))
                if rss < best_rss:
                    best_rss = rss
                    best_index = candidate_index
            if best_index is None or (current_rss - best_rss) / initial_rss < self.min_improvement:
                break
            selected.append(best_index)
            current_rss = best_rss

        self.terms_ = [candidates[index] for index in selected]
        design = np.column_stack([np.ones(len(X))] + [self._term_values(X, term) for term in self.terms_])
        self.coef_ = solve_ridge(design, y, self.alpha)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("terms_", "coef_"))
        X = check_array(X, dtype=float)
        design = np.column_stack([np.ones(len(X))] + [self._term_values(X, term) for term in self.terms_])
        return design @ self.coef_


class LocalMultilinearRegressor(RegressorMixin, BaseEstimator):
    """Local trilinear baseline suitable for CV on an incomplete training grid.

    Exact regular-grid interpolation cannot predict a grid node removed by a
    CV fold because that node's value is itself an interpolation coefficient.
    This estimator therefore performs weighted local least squares with the
    complete trilinear basis [1,x,u,f,xu,xf,uf,xuf].
    """

    def __init__(self, n_neighbors: int = 20, alpha: float = 1e-6):
        self.n_neighbors = n_neighbors
        self.alpha = alpha

    @staticmethod
    def _basis(X: np.ndarray) -> np.ndarray:
        x, u, f = X[:, 0], X[:, 1], X[:, 2]
        return np.column_stack([np.ones(len(X)), x, u, f, x * u, x * f, u * f, x * u * f])

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y, dtype=float)
        self.minimum_ = X.min(axis=0)
        span = X.max(axis=0) - self.minimum_
        self.span_ = np.where(span > 0, span, 1.0)
        self.X_ = (X - self.minimum_) / self.span_
        self.y_ = y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("X_", "y_", "minimum_", "span_"))
        X = check_array(X, dtype=float)
        queries = (X - self.minimum_) / self.span_
        predictions = []
        neighbor_count = min(max(8, self.n_neighbors), len(self.X_))
        for query in queries:
            distance = np.sqrt(np.sum((self.X_ - query) ** 2, axis=1))
            indices = np.argsort(distance)[:neighbor_count]
            local_X = self.X_[indices]
            weights = 1.0 / np.maximum(distance[indices], 0.08) ** 2
            design = self._basis(local_X)
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_y = self.y_[indices] * np.sqrt(weights)
            coefficients = solve_ridge(weighted_design, weighted_y, self.alpha)
            predictions.append((self._basis(query.reshape(1, -1)) @ coefficients).item())
        return np.asarray(predictions)


class XGBoostRegressorLite(RegressorMixin, BaseEstimator):
    """Squared-error second-order boosted trees following XGBoost equations."""

    def __init__(
        self,
        n_estimators: int = 120,
        learning_rate: float = 0.05,
        max_depth: int = 2,
        min_child_weight: float = 4.0,
        reg_lambda: float = 8.0,
        gamma: float = 0.02,
        subsample: float = 0.9,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.random_state = random_state

    def _leaf(self, gradient: np.ndarray, hessian: np.ndarray) -> dict:
        weight = -float(np.sum(gradient)) / (float(np.sum(hessian)) + self.reg_lambda)
        return {"leaf": weight}

    def _build_tree(self, X: np.ndarray, gradient: np.ndarray, hessian: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or np.sum(hessian) < 2.0 * self.min_child_weight:
            return self._leaf(gradient, hessian)
        total_gradient = float(np.sum(gradient))
        total_hessian = float(np.sum(hessian))
        parent_score = total_gradient**2 / (total_hessian + self.reg_lambda)
        best: tuple[float, int, float, np.ndarray] | None = None
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for threshold in (values[:-1] + values[1:]) / 2.0:
                left = X[:, feature] <= threshold
                left_hessian = float(np.sum(hessian[left]))
                right_hessian = total_hessian - left_hessian
                if left_hessian < self.min_child_weight or right_hessian < self.min_child_weight:
                    continue
                left_gradient = float(np.sum(gradient[left]))
                right_gradient = total_gradient - left_gradient
                gain = 0.5 * (
                    left_gradient**2 / (left_hessian + self.reg_lambda)
                    + right_gradient**2 / (right_hessian + self.reg_lambda)
                    - parent_score
                ) - self.gamma
                if best is None or gain > best[0]:
                    best = (gain, feature, float(threshold), left)
        if best is None or best[0] <= 0.0:
            return self._leaf(gradient, hessian)
        _, feature, threshold, left = best
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left], gradient[left], hessian[left], depth + 1),
            "right": self._build_tree(X[~left], gradient[~left], hessian[~left], depth + 1),
        }

    @staticmethod
    def _predict_tree(tree: dict, row: np.ndarray) -> float:
        node = tree
        while "leaf" not in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return float(node["leaf"])

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.base_score_ = float(np.mean(y))
        self.trees_: list[dict] = []
        prediction = np.full(len(y), self.base_score_)
        sample_count = min(len(y), max(2, int(math.ceil(self.subsample * len(y)))))
        for _ in range(self.n_estimators):
            gradient = prediction - y
            hessian = np.ones(len(y))
            indices = rng.choice(len(y), size=sample_count, replace=False)
            tree = self._build_tree(X[indices], gradient[indices], hessian[indices], 0)
            self.trees_.append(tree)
            prediction += self.learning_rate * np.asarray([self._predict_tree(tree, row) for row in X])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("trees_", "base_score_"))
        X = check_array(X, dtype=float)
        prediction = np.full(len(X), self.base_score_)
        for tree in self.trees_:
            prediction += self.learning_rate * np.asarray([self._predict_tree(tree, row) for row in X])
        return prediction
