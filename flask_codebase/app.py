from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import logging
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from xgboost import XGBRegressor
import math
import os
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    r2_score,
    mean_poisson_deviance,
    mean_gamma_deviance,
    accuracy_score,
)
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


API_KEY = "da7cc643495745a78c99c491e1d4d0a6"


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
app = Flask(__name__)
# Set a secret key for session management
app.secret_key = os.urandom(24)  # Generates a random 24-byte key

GLOBAL_DF_DATA = None

### Functions here ###


def fetch_historical_data(
    ticker: str, start_date: str, end_date: str, time_interval: str
) -> dict:
    data_format = "JSON"
    timezone = "utc"
    historical_data_url = f"https://api.twelvedata.com/time_series?apikey={API_KEY}&interval={time_interval}&symbol={ticker}&start_date={start_date} 16:23:00&end_date={end_date} 16:00:00&format={data_format}&timezone={timezone}"
    try:
        response = requests.get(historical_data_url)
        logging.debug(response)
        data = response.json()
        logging.info(data)
        return data
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        return None  # Return None or handle errors as needed


# Input: historical data in dictionary format
# Output: dataframe dictionary.
def data_transformation(data: dict) -> dict:
    df = pd.DataFrame(data["values"])
    # df
    # Convert specified columns to float
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    logging.info(df)
    return df


def create_dataset(dataset, time_step=3):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i : (i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


### Flask Routing here ###
@app.route("/")
def home():
    usernumber = session.get(
        "usernumber", 0
    )  # Get usernumber from session or default to 0
    usernumber += 1
    session["usernumber"] = usernumber  # Update usernumber in session
    return render_template("index.html", fig=None)  # This will render your HTML page


# this is on submit on the search for historical prices
@app.route("/submit-search-historical-prices", methods=["POST"])
def submit_search_historical_prices():
    global GLOBAL_DF_DATA
    # Get values from the form in the html page
    symbol = request.form["symbol"]
    start_date = request.form["start_date"]
    end_date = request.form["end_date"]
    time_interval = request.form["time_interval"]

    session["start_date"] = start_date
    session["end_date"] = end_date
    session["symbol"] = symbol
    # TODO: assert start_date < end_date, else display out an error.
    logging.info(
        f"CCY Pair Symbol: {symbol}, Start Date: {start_date}, End Date: {end_date}, Time Interval: {time_interval}"
    )
    historical_data = fetch_historical_data(symbol, start_date, end_date, time_interval)
    df_data = data_transformation(historical_data)

    # Display the data into html table
    price_table_html = df_data.to_html(
        classes="table table-bordered table-striped text-center",
        index=False,
        header=False,
    )

    # Store dataframe
    GLOBAL_DF_DATA = df_data
    return jsonify(price_table_html=price_table_html)


@app.route("/make-predictions", methods=["POST"])
def submit_make_predictions():
    # Get values from the form in the html page
    prediction_start_date = request.form["prediction_startDate"]
    prediction_end_date = request.form["prediction_endDate"]
    historical_data_start_date = session.get("start_date")
    historical_data_end_date = session.get("end_date")

    if historical_data_start_date > prediction_start_date:
        logging.error("Prediction start date is less than Historical start date")
        return 1
    elif historical_data_end_date < prediction_end_date:
        logging.error("Prediction end date is more than Historical start date")
        return 2

    logging.info(prediction_start_date)
    logging.info(prediction_end_date)
    global GLOBAL_DF_DATA

    numeric_df = GLOBAL_DF_DATA.select_dtypes(include=["number"])
    logging.info(numeric_df)

    # Compute mean and covariance
    df_mean = numeric_df.mean()
    logging.info(f"df_mean: {df_mean}")
    cov_returns = numeric_df.cov()
    logging.info(f"cov_returns: {cov_returns}")

    # Create a new dataframe to prepare for data training
    # Filter a smaller date time range for data training
    # GLOBAL_DF_DATA contains the whole historical data from above.
    closedf = GLOBAL_DF_DATA[["datetime", "close"]]
    logging.info("Shape of close dataframe:", closedf.shape)
    closedf = closedf[
        (closedf["datetime"] >= prediction_start_date)
        & (closedf["datetime"] <= prediction_end_date)
    ]

    close_stock = closedf.copy()
    logging.info("Total data for prediction: ", closedf.shape[0])
    del closedf["datetime"]

    # Data normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

    # Split data into traning and testing set
    # Training size 70%
    training_size = int(len(closedf) * 0.70)

    # Testing size 30%
    test_size = len(closedf) - training_size
    train_data = closedf[0:training_size, :]  # All columns for training
    test_data = closedf[training_size:, :]      # All columns for testing

    time_step = 15
    # Create dataset for training
    x_train, y_train = create_dataset(train_data, time_step)
    # Create dataset for testing
    x_test, y_test = create_dataset(test_data, time_step)

    # XGBoost regression model trained on the training dataset
    my_model = XGBRegressor(n_estimators=1000)
    my_model.fit(x_train, y_train, verbose=False)

    # Make Predictions
    predictions = my_model.predict(x_test)

    # Calculate error metrics
    
    # For MAE, lower value is better.
    logging.info(
        "Mean Absolute Error - MAE : " + str(mean_absolute_error(y_test, predictions))
    )
    
    # For RMSE, lower value is better.
    logging.info(
        "Root Mean squared Error - RMSE : "
        + str(math.sqrt(mean_squared_error(y_test, predictions)))
    )

    # Prediction on trained data (see if overfitting)
    predicions_on_trained_data = my_model.predict(x_train)

    # Reshape into 2d array
    train_predict = predicions_on_trained_data.reshape(-1, 1)
    test_predict = predictions.reshape(-1, 1)

    # Transform back to original form for display
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    # original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    # original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    look_back = time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    
    # Fill in training predictions
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # logging.info("Train predicted data: ", trainPredictPlot.shape)

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    # Fill in test predictions starting after training predictions
    # Calculate start index for test predictions
    start_index = len(train_predict) + (look_back * 2)

    # Ensure that we do not exceed the bounds of closedf
    # end_index = start_index + len(test_predict)
    
    testPredictPlot[start_index:start_index + len(test_predict), :] = test_predict
    # Fill in test predictions
    # if end_index <= len(testPredictPlot):
    #     testPredictPlot[start_index:end_index, :] = test_predict
    # else:
    #     logging.error("Mismatch in sizes: Cannot fit test predictions into plot array.")
    
    names = cycle(
        [
            "Original close price",
            "Train predicted close price",
            "Test predicted close price",
        ]
    )
    plotdf = pd.DataFrame(
        {
            "datetime": close_stock["datetime"].values,
            "original_close": close_stock["close"].values,
            "train_predicted_close": trainPredictPlot.reshape(-1).tolist(),
            "test_predicted_close": testPredictPlot.reshape(-1).tolist(),
        }
    )

    fig = px.line(
        plotdf,
        x="datetime",
        y=[
            "original_close",
            "train_predicted_close",
            "test_predicted_close",
        ],
        labels={"value": "Close price", "datetime": "Date"},
    )
    fig.update_layout(
        title_text="Comparision between original close price vs predicted close price",
        plot_bgcolor="white",
        font_size=15,
        font_color="black",
        legend_title_text="Close Price",
    )
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    #####
    x_input = test_data[len(test_data) - time_step :].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    i = 0
    pred_days = 10
    while i < pred_days:
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:])
            # logging.info("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1, -1)

            yhat = my_model.predict(x_input)
            # logging.info("{} day output {}".format(i,yhat))
            temp_input.extend(yhat.tolist())
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i = i + 1

        else:
            yhat = my_model.predict(x_input)

            temp_input.extend(yhat.tolist())
            lst_output.extend(yhat.tolist())

            i = i + 1

    logging.info("Output of predicted next days: ", len(lst_output))

    last_days = np.arange(0, time_step + 1)
    day_pred = np.arange(time_step + 1, time_step + pred_days + 1)
    logging.info(last_days)
    logging.info(day_pred)
    temp_mat = np.empty((len(last_days) + pred_days + 1, 1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1, -1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0 : time_step + 1] = (
        scaler.inverse_transform(closedf[len(closedf) - time_step :])
        .reshape(1, -1)
        .tolist()[0]
    )
    next_predicted_days_value[time_step + 1 :] = (
        scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
        .reshape(1, -1)
        .tolist()[0]
    )
    new_pred_plot = pd.DataFrame(
        {
            "last_original_days_value": last_original_days_value,
            "next_predicted_days_value": next_predicted_days_value,
        }
    )

    names = cycle(["Last 15 days close price", "Predicted next 10 days close price"])
    fig = px.line(
        new_pred_plot,
        x=new_pred_plot.index,
        y=[
            new_pred_plot["last_original_days_value"],
            new_pred_plot["next_predicted_days_value"],
        ],
        labels={"value": "Close price", "index": "Timestamp"},
    )
    fig.update_layout(
        title_text="Compare last 15 bars vs next 10 bars",
        plot_bgcolor="white",
        font_size=15,
        font_color="black",
        legend_title_text="Close Price",
    )
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # fig.show()
    # Convert the figure to JSON
    fig_json = fig.to_json()

    return jsonify(fig_json=fig_json)


if __name__ == "__main__":
    app.run(use_reloader=True, debug=True)
