from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import plotly.io as pio


pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    df = pd.read_csv(filename)  # todo maybe add skip rows, index_col as in lab 03
    df.drop(["id"], axis=1, inplace=True)
    df.drop(["date"], axis=1, inplace=True)
    df = df[df["price"] > 0]
    prices = df.loc[:, "price"]
    df.drop(["price"], axis=1, inplace=True)
    df_hot_zip = pd.get_dummies(df, columns=["zipcode"])
    return df_hot_zip, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_y = np.std(y)
    for feature in X:
        cov_feature = np.cov(X[feature], y)[0][1]
        std_feature = np.std(X[feature])
        p_corr = cov_feature / (std_feature * std_y)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X[feature],
            y=y,
            name="response as a function of {feature}.\n Corr = {correlation}".format(feature=feature,
                                                                                      correlation=p_corr),
            mode="markers",
            marker=go.scatter.Marker(
                opacity=0.6,

                colorscale="Viridis"
            )
        ))
        fig.update_layout(
            title_text="Response as function of " + str(feature) + "\n Pearson correlation= " + str(p_corr),
            yaxis_title="Response: Price", xaxis_title=str(feature))

        fig.write_image(output_path + "\{feature}_for_response.png".format(feature=feature))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset

    data_frame = load_data("house_prices.csv")
    print(data_frame)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data_frame[0], data_frame[1], "/bar/school/madmah/year2/semester_B/IML/IML.HUJI/exercises")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data_frame[0], data_frame[1], 0.75)
    print(train_X)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    percentages = np.linspace(10, 100, 91)
    all_mean_loss = []
    all_std_loss_min = []
    all_std_loss_plus = []
    for p in percentages:
        loss_list = []
        for i in range(10):
            sample_p_x = train_X.sample(frac=p / 100)
            sample_p_y = (train_y.loc[sample_p_x.index]).to_numpy()
            estimator = LinearRegression()
            np_sample = sample_p_x.to_numpy()
            estimator._fit(np_sample, sample_p_y)  # finding w_hat according to sample of p%
            cur_loss = estimator._loss(test_X.to_numpy(), test_y.to_numpy())  # calculating loss for w_hat
            loss_list.append(cur_loss)
        mean_cur_loss = np.mean(np.array(loss_list))
        all_mean_loss.append(mean_cur_loss)
        std_cur_loss = np.std(np.array(loss_list))
        all_std_loss_min.append(mean_cur_loss - (2 * std_cur_loss))
        all_std_loss_plus.append(mean_cur_loss + (2 * std_cur_loss))
    np_all_mean_loss = np.array(all_mean_loss)

    fig = go.Figure([go.Scatter(x=percentages, y=np_all_mean_loss, mode="markers"),
                     go.Scatter(x=percentages, y=np.array(all_std_loss_min),  mode="lines"),
                     go.Scatter(x=percentages, y=np.array(all_std_loss_plus),fill='tonexty', mode="lines")])
    fig.update_layout(
        title_text="Mean loss (MSE loss) as function of sample size ",
        yaxis_title="mean loss", xaxis_title="sample size")

    fig.show()
