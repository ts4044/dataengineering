# Import required libraries
import datetime
import random
import time

import pandas as pd
from polygon import RESTClient

# Importing API Key
from polygon_client import RestClient
import pymongo

"""
SQLite storage functions
"""

my_client = pymongo.MongoClient()
db = my_client["source_data"]


def reset_raw_data_tables(currency_pairs):
    """
    Reset the tables so that tables contain data for every 6 minutes only.
    """
    for curr in currency_pairs:
        collection = db[curr + "_raw"]
        collection.drop()


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


"""
Functions to calculate the metrics
"""


def aggregate_data(currency_pairs, period, n_values) -> []:
    """
    Function that runs every 6 minutes to calculate the Volatility and Keltner Bands.
    """
    keltner_bands = {}

    for curr in currency_pairs:
        # Calculates max, min and mean from raw tables
        # Using distinct ticktime
        # Cause the api sometime returns multiple values for the same tick time, not sure why
        collection = db[curr + "_raw"]
        result = collection.aggregate([{
            '$group': {
                '_id': None,
                'mean': {'$avg': '$fxrate'},
                'min': {'$min': '$fxrate'},
                'max': {'$max': '$fxrate'},
                'last_date': {'$max': '$ticktime'},

            }}])
        mean_val, volatility, fd, max_val, min_val, last_date = 0, 0, 0, 0, 0, 0
        for row in result:
            mean_val = row['mean']
            min_val = row['min']
            max_val = row['max']
            volatility = (max_val - min_val) / mean_val
            last_date = row['last_date']

        # Calculate the new Keltner Bands
        keltner_bands[curr] = get_keltner_channel(mean_val, volatility)

        fd = 0
        if volatility != 0:
            fd = n_values[curr] / volatility

        last_mean = 0
        collection = db[curr + "_agg"]
        result = collection.find(sort=[("inserttime", pymongo.DESCENDING)]).limit(1)
        for row in result:
            last_mean = row['mean']

        return_val = calculate_return(last_mean, mean_val)
        agg_data = {
            "inserttime": last_date,
            "period": period,
            "max": max_val,
            "min": min_val,
            "mean": mean_val,
            "vol": volatility,
            "fd": fd,
            "return_val": return_val,
        }
        collection.insert_one(agg_data)

    return keltner_bands


def calculate_hourly_metrics(
        currency_pairs, window, currency_check, long_currency
):
    """
    Function called every 60 minutes to make a decision on currencies that are still open for buying.
    """

    for curr in currency_pairs:
        # If currency pair is False in the currencyCheck dict the currency is closed
        if not currency_check[curr]:
            continue

        # setting position based on longCurrency list
        if curr in long_currency:
            position = "LONG"
        else:
            position = "SHORT"

        # Fetching last 10 rows from aggregate table to get the return value
        collection = db[curr + "_agg"]
        result = collection.find(sort=[("inserttime", pymongo.DESCENDING)]).limit(10)
        # return_val stores the sum of last 10 return_val
        return_val = 0
        for row in result:
            if row.return_val:
                return_val += row['return_val']

        # At T10, cutoff value to use is 0.250%
        balance = 100
        if window == 1:
            if position == "LONG":
                # Long condition -> , a profitable trade has a positive return but we have a tolerance of 0.250%
                # 0.250% = 0.0025
                if return_val >= -0.0025:
                    balance = balance + 100 + return_val
                # If not profitable, we will close the position and set the currencyCheck flag to false
                else:
                    balance = balance + return_val
                    currency_check[curr] = False
            else:
                # Short condition -> a profitable trade has a negative return.
                if return_val <= 0.0025:
                    balance = balance + 100 + return_val
                # If not profitable, we will close the position and set the currencyCheck flag to false
                else:
                    balance = balance + return_val
                    currency_check[curr] = False

        # At T20, cutoff value to use is 0.150%
        elif window == 2:
            # The below line fetches the last balance of the currency pair of last 60 min period
            collection = db[curr + "_output"]
            result = collection.find({"window": window - 1}).limit(1)
            for row in result:
                balance = row['balance']

            if position == "LONG":
                if return_val >= -0.0015:
                    balance = balance + 100 + return_val
                else:
                    balance = balance + return_val
                    currency_check[curr] = False
            else:
                # Short condition
                if return_val <= 0.0015:
                    balance = balance + 100 + return_val
                else:
                    balance = balance + return_val
                    currency_check[curr] = False

        # At T30, value to use is 0.100%
        elif window == 3:
            collection = db[curr + "_output"]
            result = collection.find({"window": window - 1}).limit(1)
            for row in result:
                balance = row['balance']

            if position == "LONG":
                if return_val >= -0.001:
                    balance = balance + 100 + return_val
                else:
                    balance = balance + return_val
                    currency_check[curr] = False
            else:
                if return_val <= 0.001:
                    balance = balance + 100 + return_val
                else:
                    balance = balance + return_val
                    currency_check[curr] = False

        # At T40, value to use is 0.050%
        elif window == 4:
            collection = db[curr + "_output"]
            result = collection.find({"window": window - 1}).limit(1)
            for row in result:
                balance = row['balance']

            if position == "LONG":
                if return_val >= -0.0005:
                    balance = balance + 100 + return_val
                else:
                    balance = balance + return_val
                    currency_check[curr] = False
            else:
                if return_val <= 0.0005:
                    balance = balance + 100 + return_val
                else:
                    balance = balance + return_val
                    currency_check[curr] = False

        # After T40, value to use is 0.050%
        elif window > 4:
            collection = db[curr + "_output"]
            result = collection.find({"window": window - 1}).limit(1)
            for row in result:
                balance = row['balance']

            if position == "LONG":
                if return_val >= -0.0005:
                    balance = balance + 100 + return_val
                else:
                    balance = balance + return_val
                    currency_check[curr] = False
            else:
                if return_val <= 0.0005:
                    balance = balance + 100 + return_val
                else:
                    balance = balance + return_val
                    currency_check[curr] = False

        output_data = {
            "window": window,
            "balance": balance,
            "return_val": return_val,
            "position": position,
        }
        collection.insert_one(output_data)

    return currency_check


def main(currency_pairs):
    """
    Repeatedly calls the polygon api every 1 seconds for 24 hours.
    Data is added to *_raw tables in SQLite.
    Every 6 minutes, calculates the new Keltner Bands using the raw tables.
    Fractal Dimension, Volatility and Returns are calculated based on the new Keltner bands
     and added to aggregate table.

    Every 1 hour, use estimates and errors to make decision whether to buy more currency,
        do nothing or stop buying the currency.
    If the estimates and errors are in line - buy more currency;
    If estimates are pointing towards decline and errors are negative or visa versa, do nothing;
    If estimates and errors are pointing to decline, stop buying more currency.
    """
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

    # Long currencies
    long_currency = random.sample(currency_pairs, k=4)
    # Short Currencies
    short_currency = [x for x in currency_pairs if x not in long_currency]

    for currency in currency_pairs:
        n_values[currency] = 0
        currency_check[currency] = True

    # Loop that runs until the total duration of the program hits 24 hours.
    while counter < 72010:

        if counter % 360 == 0:
            period_count += 1
            print(period_count, n_values)

            # Calculate the keltner bands
            keltner_bands = aggregate_data(
                currency_pairs, period_count, n_values
            )
            reset_raw_data_tables(currency_pairs)

            if counter % 3600 == 0:
                print(period_count, currency_check)
                window = period_count / 10
                currency_check = calculate_hourly_metrics(
                    currency_pairs, window, currency_check, long_currency
                )

            n_values = {}
            last_price = {}
        else:
            # Only call the API every 1 second
            time.sleep(0.65)

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
            raw_data = {
                "ticktime": dt,
                "fxrate": avg_price,
                "inserttime": insert_time,
            }
            collection = db[currency + "_raw"]
            collection.insert_one(raw_data)

        # Increment the counters
        counter += 1


# def generate_csv():
#     # Code to convert db tables into csv files
#     conn = sqlite3.connect(
#         "sqlite/prediction.db", isolation_level=None, detect_types=sqlite3.PARSE_COLNAMES
#     )
#
#     for curr in currencies:
#         sql = "SELECT * FROM " + curr + "_agg"
#         db_df = pd.read_sql_query(sql, conn)
#         path = "data/" + curr + ".csv"
#         db_df.to_csv(path, index=False)
#
#     for curr in currencies:
#         sql = "SELECT * FROM " + curr + "_output"
#         db_df = pd.read_sql_query(sql, conn)
#         path = "output/" + curr + ".csv"
#         db_df.to_csv(path, index=False)


# A dictionary defining the set of currency pairs we will be pulling data for
currencies = [
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
main(currencies)

# Generate the CSVs required
# generate_csv()
