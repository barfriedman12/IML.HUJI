# import IMLearn.learners.regressors.linear_regression
# from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df["Year"] = df["Year"].astype(str)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df = df[df["Temp"] > -70]

    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("City_Temperature.csv")
    print(x)
    # Question 2 - Exploring data for specific country
    israel_df = X.loc[X["Country"] == "Israel", :]
    fig_temps_day_of_year = px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year", title="Temp as function of day of a year").show()

    temp_std_by_month = israel_df.groupby("Month").agg({"Temp": "std"})
    print(temp_std_by_month)

    fig_std_month_bar = px.bar(temp_std_by_month, y="Temp",title="Monthly std of temp").show()

    # Question 3 - Exploring differences between countries
    country_month_temp = X.groupby(["Country", "Month"]).agg({"Temp": ["mean", "std"]})
    country_month_temp.columns = ["mean_temp", "std_temp"]
    print(country_month_temp)
    country_month_temp = country_month_temp.reset_index()
    print("after reset=", country_month_temp)
    fig_avg_temp = px.line(country_month_temp,x= "Month",y= "mean_temp",color = "Country",error_y = "std_temp",title="Monthly average temp per country").show()

    temp_israel = israel_df.loc[:, "Temp"]
    day_of_year_israel = israel_df.loc[:, "DayOfYear"]

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(day_of_year_israel, temp_israel, 0.75)
    all_losses = []
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(train_X.to_numpy(), train_y.to_numpy())
        loss_k = poly_fit.loss(test_X.to_numpy(), test_y.to_numpy())
        all_losses.append(round(loss_k, 2))
    print(all_losses)
    plot_df = pd.DataFrame({"k": range(1, 11), "Loss": all_losses})
    fig_error_per_k = px.bar(plot_df, x="k", y="Loss", title="Loss per k").show()

    # Question 5 - Evaluating fitted model on different countries
    # # k=5
    poly_fit_israel_temp = PolynomialFitting(5)
    poly_fit_israel_temp.fit(day_of_year_israel.to_numpy(), temp_israel.to_numpy())
    all_errs = []
    countries = ["Jordan", "South Africa","The Netherlands"]
    for country in countries:
        cur_country_df = X.loc[X["Country"] == country, :]
        cur_country_day_of_year = cur_country_df.loc[:, "DayOfYear"]
        cur_country_temp = cur_country_df.loc[:, "Temp"]
        loss_k = poly_fit_israel_temp.loss(cur_country_day_of_year.to_numpy(), cur_country_temp.to_numpy())
        all_errs.append(loss_k)
    fig_error_per_country = px.bar(y=np.array(all_errs), x=np.array(countries), labels={"x": "Countries","y": " Error"},
                                   title="Israel's model error for other countries").show()
