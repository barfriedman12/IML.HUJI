import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    n_samples = y_true.shape[0]
    sqr_err = np.linalg.norm(y_true - y_pred) ** 2
    mse = (1 / n_samples) * sqr_err
    return mse


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    n_samples = y_true.shape[0]
    sum_of_errs = 0
    for i in range(n_samples):
        if y_true[i] != y_pred[i]:
            sum_of_errs += 1
    if normalize:
        return sum_of_errs / n_samples
    return sum_of_errs


        # return abs(sum(((y_true + 1) / 2) @ ((y_pred + 1) / 2))) / n_samples
    # return abs(sum(((y_true + 1) / 2) * ((y_pred + 1) / 2)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """

    # accuracy = (TP+TN)/(P+N) = T/n_samples
    n_samples = y_true.shape[0]
    num_of_t = 0
    for i in range(n_samples):
        if y_true[i] == y_pred[i]:
            num_of_t +=1
    return num_of_t/n_samples




def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()


y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
y_pred = np.array([199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])

print(mean_square_error(y_true, y_pred))
