from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from IMLearn.metrics import accuracy
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2, pi

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    nd_array = np.load(filename)
    X = nd_array[:, 0:2]
    y = nd_array[:, 2]
    return X, y


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        cur_X, cur_y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        callback = lambda perc, nothing_X, nothing_y: (perc.loss(cur_X, cur_y))
        my_perc = Perceptron(callback=callback)
        my_perc.fit(cur_X, cur_y)
        losses = np.array(my_perc.training_loss_)
        num_iters = np.arange(start=0, stop=my_perc.num_iters)
        #
        fig = go.Figure([go.Scatter(x=num_iters, y=losses, mode="markers+lines")])
        fig.update_layout(
            title_text="Misclassification error loss of  {} data as function of Perceptron iteration".format(n),
            yaxis_title="Loss", xaxis_title="Iteration")
        # fig.show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    symbols = np.array(["circle", "x"])
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        cur_X, cur_y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(cur_X, cur_y)
        y_pred_lda = lda.predict(cur_X)


        gaus_naive = GaussianNaiveBayes()
        gaus_naive.fit(cur_X, cur_y)
        y_pred_gaus_naive = gaus_naive.predict(cur_X)
        print("y_pred_gaus_naive=",y_pred_gaus_naive)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[
                                f'Gaussian Naive Bayes predictions '
                                f'with accuracy {accuracy(cur_y, y_pred_gaus_naive)}',
                                f'LDA predictions with accuracy'
                                f' {accuracy(cur_y, y_pred_lda)}'])

        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(x=cur_X[:, 0], y=cur_X[:, 1], showlegend=False,
                                 mode='markers', marker=dict(color=y_pred_gaus_naive, symbol=cur_y)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=cur_X[:, 0], y=cur_X[:, 1], showlegend=False,
                                 mode='markers', marker=dict(color=y_pred_lda, symbol=cur_y)),
                      row=1, col=2)
        fig.update_xaxes(title_text='feature 1', row=1, col=1)
        fig.update_xaxes(title_text='feature 1', row=1, col=2)
        fig.update_yaxes(title_text='feature 2', row=1, col=1)
        fig.update_yaxes(title_text='feature 2', row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(3):
            fig.add_trace(go.Scatter(x=[gaus_naive.mu_[i][0]], y=[gaus_naive.mu_[i][1]],
                                     showlegend=False, mode='markers', marker=dict(color='black',
                                                                                   symbol='cross', size=8)), row=1,
                          col=1)
            fig.add_trace(go.Scatter(x=[lda.mu_[i][0]], y=[lda.mu_[i][1]],
                                     showlegend=False, mode='markers', marker=dict(color='black',
                                                                                   symbol='cross', size=8)), row=1,
                          col=2)
            fig.add_trace(get_ellipse(gaus_naive.mu_[i], np.diag(gaus_naive.vars_[i])), row=1, col=1)
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)
        fig.show()


if __name__ == '__main__':
    # np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()
    data = np.array(((0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2), (7, 2)))
    X = data[0]
    y = data[1]
    gaus_naive = GaussianNaiveBayes()
    gaus_naive.fit(X, y)
    y_pred_gaus_naive = gaus_naive.predict(X)
    print(y_pred_gaus_naive)

