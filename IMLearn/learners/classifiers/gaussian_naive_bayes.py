from typing import NoReturn
from ...base import BaseEstimator
from IMLearn.metrics import loss_functions
import numpy as np
import pandas as pd


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """


        self.classes_, n_k_vec = np.unique(y, return_counts=True)
        n_classes, n_samples, n_features = self.classes_.shape[0], X.shape[0], X.shape[1]
        self.pi_ = n_k_vec / n_samples
        self.mu_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))

        for k in range(n_classes):
            x_k = X[y == self.classes_[k]]
            sum_mu_vec_k = np.sum(x_k, axis=0)
            self.mu_[k] = sum_mu_vec_k / n_k_vec[k]
            sum_var_vec_k = np.sum((x_k - self.mu_[k]) ** 2,axis=0)
            self.vars_[k] = sum_var_vec_k / n_k_vec[k]



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
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_samples, n_features = X.shape
        n_classes = self.classes_.shape[0]
        const = ((-n_features) / 2) * np.log(2 * np.pi)
        log_likelihoods = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            all_i_likelihoods = np.zeros((n_classes,))
            for k in range(n_classes):
                log_det_cov = np.log(np.prod(self.vars_[k, :]))
                mu_k = self.mu_[k]
                pi_k = np.log(self.pi_[k])
                in_exp = -0.5 * ((X[i, :] - mu_k).T @ np.diag(1 / self.vars_[k]) @ (X[i, :] - mu_k))
                all_i_likelihoods[k] = const + in_exp + log_det_cov + pi_k
            log_likelihoods[i] = all_i_likelihoods

        return log_likelihoods

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

