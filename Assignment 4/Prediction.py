#!/usr/bin/env python
# coding: utf-8

# # HW 4 - Prediction
# 
# Here, we predict the return value based on the models we created in the previous step


# Install polygon-api-client library
# get_ipython().system('pip install polygon-api-client')
# get_ipython().system('pip install pycaret')


# Import required libraries
import csv
import datetime
import sqlite3
import time

import numpy as np
import pandas as pd
from numpy import mean
from polygon import RESTClient

# Importing API Key
from polygon_client import RestClient
from pycaret.regression import *
from sqlalchemy import create_engine, text

"""
SQLite storage functions
"""


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
                    + "_agg (inserttime text, period numeric, max numeric, min numeric, "
                    + "mean numeric, vol numeric, fd numeric, return_val numeric);"
                )
            )


"""
Functions that perform Calculations
From HW 2 and 3
"""


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


def aggregate_data(engine, currency_pairs, period, n_values):
    """
    Function that runs every 6 minutes to calculate the Volatility and Keltner Bands.
    """
    keltner_bands = {}

    with engine.begin() as conn:
        for curr in currency_pairs:
            # Calculates max, min and mean from raw tables
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
                volatility = max_val - min_val

            # Calculate the new Keltner Bands
            keltner_bands[curr] = get_keltner_channel(mean_val, volatility)

            date_res = conn.execute(
                text("SELECT MAX(ticktime) as last_date FROM " + curr + "_raw;")
            )
            for row in date_res:
                last_date = row.last_date

            if volatility == 0:
                fd = 0
            else:
                fd = n_values[curr] / volatility

            last_mean = 0
            result = conn.execute(
                text("SELECT * FROM " + curr + "_agg ORDER BY rowid DESC LIMIT 1;")
            )
            for row in result:
                last_mean = row.mean

            return_val = calculate_return(last_mean, mean_val)

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


def main(currency_pairs):
    """
    Repeatedly calls the polygon api every 1 seconds for 24 hours.
    Data is added to *_raw tables in SQLite.
    Every 6 minutes, calculates the new Keltner Bands using the raw tables.
    Fractal Dimension, Volatility and Returns are calculated based on the new Keltner bands and added to aggregate table.
    """
    # Create an engine to connect to the database; setting echo to false should stop it from logging in std.out
    engine = create_engine("sqlite:///predict.db", echo=False, future=True)
    # Initialize database
    initialize_raw_data_tables(engine, currency_pairs)
    initialize_aggregate_tables(engine, currency_pairs)

    # Get the API key from the library
    key = RestClient.fetch_key()
    # Open a RESTClient for making the api calls
    client = RESTClient(key)

    # Initializing the in-memory data structures
    counter = 0
    six_min_counter = 0
    period_count = 0
    n_values = {}
    keltner_bands = {}
    last_price = {}

    for currency in currency_pairs:
        n_values[currency] = 0

    # Loop that runs until the total duration of the program hits 24 hours.
    while counter < 36010:

        if six_min_counter == 360:
            period_count += 1
            # Calculate the keltner bands
            keltner_bands = aggregate_data(
                engine, currency_pairs, period_count, n_values
            )
            reset_raw_data_tables(engine, currency_pairs)

            six_min_counter = 0
            n_values = {}
            last_price = {}

        # Increment the counters
        counter += 1
        six_min_counter += 1

        # Only call the API every 1 second
        time.sleep(0.75)

        for currency in currency_pairs:
            # Call the API with the required parameters
            try:
                from_ = currency[:3]
                to = currency[3:]
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


def generate_csv():
    # Code to convert db tables into csv files
    conn = sqlite3.connect(
        "predict.db", isolation_level=None, detect_types=sqlite3.PARSE_COLNAMES
    )

    for curr in currency_pairs:
        sql = "SELECT * FROM " + curr + "_agg"
        db_df = pd.read_sql_query(sql, conn)
        path = "results/" + curr + ".csv"
        db_df.to_csv(path, index=False)


"""
Functions for predictions
"""


def load_saved_models(currency_pairs):
    models = {}
    for currency in currency_pairs:
        models[currency] = load_model("./models/{}".format(currency))
    return models


def load_divider_values(currency_pairs):
    cutoff_values = {}
    for currency in currency_pairs:
        cutoff_values[currency] = {}

    with open("./regression_code/divider_list.csv", "r") as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)

        for row in csvreader:
            cutoff_values[row[0]][row[1]] = [float(row[2]), float(row[3])]

    return cutoff_values


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

# # Load the saved models into a dictionary
# saved_models = load_saved_models(currency_pairs)
#
# # Load the cutoff values into a dictionary
# cutoff_divider_values = load_divider_values(currency_pairs)
#
# # Generate the CSVs required
# generate_csv()
