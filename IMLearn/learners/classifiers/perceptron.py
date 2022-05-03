from __future__ import annotations
from typing import Callable
from typing import NoReturn
from IMLearn.metrics import loss_functions
from ...base import BaseEstimator
import numpy as np


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    training_loss_: array of floats
        holds the loss value of the algorithm during training.
        training_loss_[i] is the loss value of the i'th training iteration.
        to be filled in `Perceptron.fit` function.

    """

    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        max_iter): int, default = 1000
            Maximum number of passes over training data

        callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by Perceptron. To be set in `Perceptron.fit` function.
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None
        self.training_loss_ = []
        self.num_iters = max_iter

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        nothing_x, nothing_y = X, 0

        n_samples, n_features = X.shape
        self.training_loss_ = []
        if self.include_intercept_:
            x0 = np.ones((n_samples, 1))
            X = np.hstack((x0, X))
            n_features += 1
        self.coefs_ = np.zeros(n_features, )
        for t in range(1, self.max_iter_):
            is_w_good = True
            for i in range(n_samples):
                if y[i] * (self.coefs_.T @ X[i]) <= 0:
                    is_w_good = False
                    self.coefs_ += y[i] * X[i]
                    self.fitted_ = True
                    self.training_loss_.append(self.callback_(self, nothing_x, nothing_y))
                    break
            if is_w_good:
                self.num_iters = t
                break

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y = []
        n_samples = X.shape[0]
        if self.include_intercept_:
            x0 = np.ones((n_samples, 1))
            X = np.hstack((x0, X))
        for i in range(n_samples):
            y.append(np.sign(self.coefs_.T @ X[i, :]))
        return np.array(y)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        loss = loss_functions.misclassification_error(y, y_pred)
        return loss
