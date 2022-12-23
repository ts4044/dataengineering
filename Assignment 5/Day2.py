#!/usr/bin/env python
# coding: utf-8

# # Day 2 - Real Time Stop Loss based on Predictions

# In[ ]:


# Import required libraries
import csv
import datetime
import random
import sqlite3
import time

import pandas as pd
from numpy import mean
from polygon import RESTClient
from pycaret.regression import *
from sqlalchemy import create_engine, text

from polygon_client import RestClient


# ### SQLite storage functions

# In[ ]:


def reset_raw_data_tables(engine, currency_pairs):
    """
    Reset the tables so that tables contain data for every 6 minutes only.
    """
    with engine.begin() as conn:
        for curr in currency_pairs:
            conn.execute(text("DROP TABLE IF EXISTS " + curr + "_raw;"))
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS "
                    + curr
                    + "_raw(ticktime text, fxrate  numeric, inserttime text);"
                )
            )


def initialize_raw_data_tables(engine, currency_pairs):
    """
    Creates tables to store the raw data from polygon api
    """
    with engine.begin() as conn:
        for curr in currency_pairs:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS "
                    + curr
                    + "_raw(ticktime text, fxrate  numeric, inserttime text);"
                )
            )


def initialize_aggregate_tables(engine, currency_pairs):
    """
    Create a table for storing the aggregated data for each currency pair
    """
    with engine.begin() as conn:
        for curr in currency_pairs:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS "
                    + curr
                    + "_agg (inserttime text, period numeric, max numeric, "
                    + "min numeric, mean numeric, vol numeric, fd numeric, return_val numeric);"
                )
            )


def initialize_output_tables(engine, currency_pairs):
    """
    Create a table to store the balance and return for every hour
    """
    with engine.begin() as conn:
        for curr in currency_pairs:
            conn.execute(
                text(
                    "CREATE TABLE "
                    + curr
                    + "_output (window numeric, balance numeric, return_val numeric,"
                      " predicted_return numeric, position text);"
                )
            )


# ### Utility functions for calculations

# In[ ]:


def get_keltner_channel(mean_val, vol) -> []:
    """
    Function to calculate Keltner Bands
    """
    keltner_upper_band = []
    keltner_lower_band = []

    for i in range(1, 101):
        keltner_upper_band.append(mean_val + (i * 0.025 * vol))
        keltner_lower_band.append(mean_val - (i * 0.025 * vol))

    # Reversing the Keltner Lower Band result and appending it to Upper Band Result
    # This sorts the final result from the lowest band to the highest band
    result = keltner_lower_band[::-1]
    result.extend(keltner_upper_band)
    return result


def calculate_keltner_intersection(last_price, current_price, kbands) -> int:
    """
    This calculates the total number of times the bands were crossed for two values.
    """
    count = 0

    # For each band value, check if the line from previous value to current value
    # crosses the keltner band. If yes, increase count
    for band in kbands:
        if last_price < band < current_price:
            count += 1
        elif last_price > band > current_price:
            count += 1

    return count


def calculate_return(last_price, current_price) -> float:
    """
    Calculate the return using the formula 𝑟𝑖 = (𝑃𝑖 − 𝑃𝑖−1)⁄(𝑃𝑖−1)
    """
    if last_price == 0:
        return 0
    return (current_price - last_price) / last_price


def timestamp_to_datetime(ts) -> str:
    return datetime.datetime.fromtimestamp(ts / 1000.0).strftime("%Y-%m-%d %H:%M:%S")


# ### Functions for predictions

# In[ ]:


def load_saved_models(currency_pairs):
    """
    Function to load saved models to use for predicting return values
    """

    models = {}
    for currency in currency_pairs:
        models[currency] = load_model("./models/{}".format(currency))
    return models


def load_divider_values(currency_pairs):
    """
    Function to load divider values to classify the column values
    """

    cutoff_values = {}
    for currency in currency_pairs:
        cutoff_values[currency] = {}

    with open("./divider_list.csv", "r") as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)

        for row in csvreader:
            try:
                cutoff_values[row[0]][row[1]] = [float(row[2]), float(row[3])]
            except:
                pass

    return cutoff_values


def substitute_cutoffs(df, cutoff_values, column):
    """
    Function to classify column values into buckets
    """

    for index, row in df.iterrows():
        # If the column value is grater than the first cutoff value, set it to 1 = High values
        if row[column] > cutoff_values[column][0]:
            df.at[index, column] = 1
        # Else If the column value is between the cutoff values, set it to 2 = Medium values
        elif cutoff_values[column][0] < row[column] < cutoff_values[column][1]:
            df.at[index, column] = 2
        # Else column value is lesser than lower cutoff value, set it to 3 = Low values
        else:
            df.at[index, column] = 3
    return df


# ### Functions to calculate the metrics at 6 minutes or 1 hour

# In[ ]:


def aggregate_data(engine, currency_pairs, period, n_values) -> []:
    """
    Function that runs every 6 minutes to calculate the Volatility and Keltner Bands.
    """
    keltner_bands = {}

    with engine.begin() as conn:
        for curr in currency_pairs:
            # Calculates max, min and mean from raw tables
            # Using distinct ticktime, cause the api sometime returns
            # multiple values for the same tick time, not sure why
            result = conn.execute(
                text(
                    "SELECT AVG(fxrate) as mean_val, "
                    "MIN(fxrate) as min_val, MAX(fxrate) as max_val FROM "
                    "(SELECT DISTINCT ticktime, fxrate from " + curr + "_raw);"
                )
            )

            for row in result:
                mean_val = row.mean_val
                min_val = row.min_val
                max_val = row.max_val
                volatility = (max_val - min_val) / mean_val

            # Calculate the new Keltner Bands
            keltner_bands[curr] = get_keltner_channel(mean_val, volatility)

            # Get the maximum timestamp to use for aggregate insertion
            date_res = conn.execute(
                text("SELECT MAX(ticktime) as last_date FROM " + curr + "_raw;")
            )
            for row in date_res:
                last_date = row.last_date

            # Calculate Fractal Dimension using the formul
            fd = 0
            if volatility != 0:
                fd = n_values[curr] / volatility

            # Get the previous mean value
            last_mean = 0
            result = conn.execute(
                text("SELECT * FROM " + curr + "_agg ORDER BY rowid DESC LIMIT 1;")
            )
            for row in result:
                last_mean = row.mean

            # Calculate the return value
            return_val = calculate_return(last_mean, mean_val)

            # Insert the results into aggregation table
            conn.execute(
                text(
                    "INSERT INTO "
                    + curr
                    + "_agg (inserttime, period, max, min, mean, vol, fd, return_val)"
                    + "VALUES (:inserttime, :period, :max, :min, :mean, :vol, :fd, :return_val );"
                ),
                [
                    {
                        "inserttime": last_date,
                        "period": period,
                        "max": max_val,
                        "min": min_val,
                        "mean": mean_val,
                        "vol": volatility,
                        "fd": fd,
                        "return_val": return_val,
                    }
                ],
            )

    return keltner_bands


def calculate_hourly_metrics(
        engine, currency_pairs, window, currencyCheck, long_currency
):
    """
    Function called every 60 minutes to make a decision on currencies that are still open for buying.
    """

    # Load the saved models into a dictionary
    saved_models = load_saved_models(currency_pairs)
    # Load the cutoff values into a dictionary
    cutoff_divider_values = load_divider_values(currency_pairs)

    for curr in currency_pairs:
        # If currency pair is False in the currencyCheck dict the currency is closed
        if not currencyCheck[curr]:
            continue

        # setting position based on longCurrency list
        if curr in long_currency:
            position = "LONG"
        else:
            position = "SHORT"

        with engine.begin() as conn:
            # Fetching last 10 rows from aggregate table to get the return value
            sql = "SELECT *  FROM " + curr + "_agg ORDER BY rowid DESC LIMIT 10"
            df = pd.read_sql_query(text(sql), conn)

            # return_val stores the actual sum of last 10 return_val
            return_val = df["return_val"].sum()

            # Predict the returns for the past one hour
            df = substitute_cutoffs(df, cutoff_divider_values[curr], "vol")
            df = substitute_cutoffs(df, cutoff_divider_values[curr], "fd")
            # Value prediction
            prediction = predict_model(saved_models[curr], df)
            # Correct the columns since we multiplied it by 100000 when creating model
            prediction.Label /= 100000
            prediction.rename(columns={"Label": "predicted"}, inplace=True)

            # predicted_return stores the predicted sum of last 10 return_val
            predicted_return = prediction["predicted"].sum()

            # At T10, cutoff value to use is 0.250% = 0.0025
            if window == 1:
                balance = 100
                if position == "LONG":

                    # Long condition - a profitable trade has a positive return
                    # If the predicted value and actual value align, invest
                    if return_val >= -0.0025 and predicted_return >= -0.0025:
                        balance = balance + 100 + return_val
                    # If small error, do nothing
                    elif (return_val >= -0.0025 and predicted_return <= 0.0025) or (
                            return_val <= 0.0025 < predicted_return
                    ):
                        balance = balance + return_val
                    # Else stop investing
                    else:
                        balance = balance + return_val
                        currencyCheck[curr] = False
                else:
                    # Short condition - a profitable trade has a negative return.
                    if return_val <= 0.0025 and predicted_return < -0.0025:
                        balance = balance + 100 + return_val
                    elif (return_val <= 0.0025 and predicted_return > -0.0025) or (
                            return_val > -0.0025 and predicted_return < 0.0025
                    ):
                        balance = balance + return_val
                    else:
                        balance = balance + return_val
                        currencyCheck[curr] = False

            # At T20, cutoff value to use is 0.150% = 0.0015
            elif window == 2:
                # The below line fetches the last balance of the currency pair of last 60 min period
                result = conn.execute(
                    text(
                        "SELECT * FROM "
                        + curr
                        + "_output WHERE window = "
                        + str(window - 1)
                        + ";"
                    )
                )
                for row in result:
                    balance = row.balance

                if position == "LONG":
                    if return_val >= -0.0015 and predicted_return >= -0.0015:
                        balance = balance + 100 + return_val
                    elif (return_val >= -0.0015 and predicted_return <= 0.0015) or (
                            return_val <= 0.0015 < predicted_return
                    ):
                        balance = balance + return_val
                    else:
                        balance = balance + return_val
                        currencyCheck[curr] = False
                else:
                    if return_val <= 0.0015 and predicted_return < -0.0015:
                        balance = balance + 100 + return_val
                    elif (return_val <= 0.0015 and predicted_return > -0.0015) or (
                            return_val > -0.0015 and predicted_return < 0.0015
                    ):
                        balance = balance + return_val
                    else:
                        balance = balance + return_val
                        currencyCheck[curr] = False

            # At T30, value to use is 0.100% = 0.001
            elif window == 3:
                result = conn.execute(
                    text(
                        "SELECT * FROM "
                        + curr
                        + "_output WHERE window = "
                        + str(window - 1)
                        + ";"
                    )
                )
                for row in result:
                    balance = row.balance

                if position == "LONG":
                    if return_val >= -0.001 and predicted_return >= -0.001:
                        balance = balance + 100 + return_val
                    elif (return_val >= -0.001 and predicted_return <= 0.001) or (
                            return_val <= 0.001 < predicted_return
                    ):
                        balance = balance + return_val
                    else:
                        balance = balance + return_val
                        currencyCheck[curr] = False
                else:
                    if return_val <= 0.001 and predicted_return < -0.001:
                        balance = balance + 100 + return_val
                    elif (return_val <= 0.001 and predicted_return > -0.001) or (
                            return_val > -0.001 and predicted_return < 0.001
                    ):
                        balance = balance + return_val
                    else:
                        balance = balance + return_val
                        currencyCheck[curr] = False

            # At and beyond T40, value to use is 0.050%
            elif window >= 4:
                result = conn.execute(
                    text(
                        "SELECT * FROM "
                        + curr
                        + "_output WHERE window = "
                        + str(window - 1)
                        + ";"
                    )
                )
                for row in result:
                    balance = row.balance

                if position == "LONG":
                    if return_val >= -0.0005 and predicted_return >= -0.0005:
                        balance = balance + 100 + return_val
                    elif (return_val >= -0.0005 and predicted_return <= 0.0005) or (
                            return_val <= 0.0005 < predicted_return
                    ):
                        balance = balance + return_val
                    else:
                        balance = balance + return_val
                        currencyCheck[curr] = False
                else:
                    if return_val <= 0.0005 and predicted_return < -0.0005:
                        balance = balance + 100 + return_val
                    elif (return_val <= 0.0005 and predicted_return > -0.0005) or (
                            return_val > -0.0005 and predicted_return < 0.0005
                    ):
                        balance = balance + return_val
                    else:
                        balance = balance + return_val
                        currencyCheck[curr] = False

            # Store the output in the output table
            conn.execute(
                text(
                    "INSERT INTO "
                    + curr
                    + "_output (window, balance, return_val, predicted_return, position) VALUES (:window, :balance, "
                      ":return_val, :predicted_return, :position); "
                ),
                [
                    {
                        "window": window,
                        "balance": balance,
                        "return_val": return_val,
                        "predicted_return": predicted_return,
                        "position": position,
                    }
                ],
            )

    return currencyCheck


# ### Main Function

# In[ ]:


def main(currency_pairs):
    """
    Repeatedly calls the polygon api every 1 seconds for 24 hours.
    Data is added to *_raw tables in SQLite.
    Every 6 minutes, calculates the new Keltner Bands using the raw tables.
    Fractal Dimension, Volatility and Returns are calculated based on the new Keltner bands and added to aggregate table.

    Every 1 hour, use estimates and errors to make decision whether to buy more currency, do nothing or stop buying the currency.
    If the estimates and errors are in line - buy more currency;
    If estimates are pointing towards decline and errors are negative or visa versa, do nothing;
    If estimates and errors are pointing to decline, stop buying more currency.
    """
    # Create an engine to connect to the database; setting echo to false should stop it from logging in std.out
    # Initialize database
    engine = create_engine("sqlite:///./sqlite/day2.db", echo=False, future=True)
    initialize_raw_data_tables(engine, currency_pairs)
    initialize_aggregate_tables(engine, currency_pairs)
    initialize_output_tables(engine, currency_pairs)

    # Get the API key from the library
    key = RestClient.fetch_key()
    # Open a RESTClient for making the api calls
    client = RESTClient(key)

    # Initializing the in-memory data structures
    counter = 1
    period_count = 0
    n_values = {}
    keltner_bands = {}
    last_price = {}
    currency_check = {}

    for currency in currency_pairs:
        n_values[currency] = 0
        currency_check[currency] = True

    # Long currencies
    long_currency = random.choices(currency_pairs, k=5)
    # Short Currencies
    short_currency = [x for x in currency_pairs if x not in long_currency]

    # Loop that runs until the total duration of the program hits 24 hours.
    while counter < 36010:
        # Every six minutes - 360 seconds.
        if counter % 360 == 0:
            period_count += 1

            # Calculate the new keltner bands and also aggregate the data into mean, vol, fd and returns.
            keltner_bands = aggregate_data(
                engine, currency_pairs, period_count, n_values
            )
            # Reset the raw data table so that it hold only data for the next 6 minute window 
            reset_raw_data_tables(engine, currency_pairs)

            # Every one hour - 3600 seconds
            if counter % 3600 == 0:
                window = period_count / 10
                calculate_hourly_metrics(
                    engine, currency_pairs, window, currency_check, long_currency
                )

            # Reset the n_values and last price dictionaries so that they hold new values for the next window
            n_values = {}
            last_price = {}
        else:
            # Only call the API every 1 second, hence wait here for 0.65s.
            # Code runs for 0.35 seconds normally and 1.2 seconds when the aggregation functions are called.
            time.sleep(0.65)

        for currency in currency_pairs:
            # Call the API with the required parameters
            try:
                # Eg, curr = USDSGD. curr[:3] = USD, curr[3:] = SGD
                from_ = currency[:3]
                to = currency[3:]
                # Plygon API call
                resp = client.forex_currencies_real_time_currency_conversion(
                    from_, to, amount=100, precision=2
                )
            except:
                continue

            # This gets the Last Trade object defined in the API Resource
            last_trade = resp.last

            # Format the timestamp from the result
            dt = timestamp_to_datetime(last_trade['timestamp'])

            # Get the current time and format it
            insert_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Calculate the price by taking the average of the bid and ask prices
            avg_price = (last_trade['bid'] + last_trade['ask']) / 2

            if keltner_bands != {}:
                if currency in last_price:
                    n_values[currency] += calculate_keltner_intersection(
                        last_price[currency],
                        avg_price,
                        keltner_bands[currency],
                    )
                    last_price[currency] = avg_price
                else:
                    last_price[currency] = avg_price
                    n_values[currency] = 0

            # Write the data to the SQLite database, raw data tables
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO "
                        + from_
                        + to
                        + "_raw(ticktime, fxrate, inserttime) VALUES (:ticktime, :fxrate, :inserttime)"
                    ),
                    [
                        {
                            "ticktime": dt,
                            "fxrate": avg_price,
                            "inserttime": insert_time,
                        }
                    ],
                )
        # Increment the counters
        counter += 1


# ### Function to generate CSV

# In[ ]:


def generate_csv():
    # Code to convert db tables into csv files
    conn = sqlite3.connect(
        "./sqlite/day2.db",
        isolation_level=None,
        detect_types=sqlite3.PARSE_COLNAMES,
    )

    for curr in currency_pairs:
        sql = "SELECT * FROM " + curr + "_agg"
        db_df = pd.read_sql_query(sql, conn)
        path = "results/" + curr + ".csv"
        db_df.to_csv(path, index=False)

    for curr in currency_pairs:
        sql = "SELECT * FROM " + curr + "_output"
        db_df = pd.read_sql_query(sql, conn)
        path = "results/" + curr + ".csv"
        db_df.to_csv(path, index=False)


# ### Invoking the functions

# In[ ]:


# A dictionary defining the set of currency pairs we will be pulling data for
currency_pairs = [
    "EURUSD",
    "GBPUSD",
    "USDCAD",
    "USDCHF",
    "USDHKD",
    "USDAUD",
    "USDNZD",
    "USDSGD",
]

# Run the main data collection loop
main(currency_pairs)

# Generate the CSVs required
generate_csv()
