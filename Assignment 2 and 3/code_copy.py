import csv
import time
import random

from polygon import RESTClient
from polygon_client import RestClient
from statistics import mean

stop_loss_values = [0, 0.25, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
# A dictionary defining the set of currency pairs we will be pulling data for
pairs = [
    ["AUD", "USD"],
    ["GBP", "EUR"],
    ["USD", "CAD"],
    ["USD", "JPY"],
    ["USD", "MXN"],
    ["EUR", "USD"],
    ["EUR", "JPY"],
    ["USD", "CZK"],
    ["USD", "PLN"],
    ["USD", "INR"],
]
pair_names = ["_".join(x) for x in pairs]
long_trades = random.choices(pair_names)
short_trades = [x for x in pair_names if x not in long_trades]


# This function calculates the keltner bands using the EMA and Volatility
def get_keltner_channel(mean_val, vol):
    keltner_upper_band = []
    keltner_lower_band = []

    for i in range(1, 101):
        keltner_upper_band.append(round(mean_val + (i * 0.025 * vol), 7))
        keltner_lower_band.append(round(mean_val - (i * 0.025 * vol), 7))

    # Reversing the Keltner Lower Band result and appending it to Upper Band Result
    # This sorts the final result from lowest band to highest band
    result = keltner_lower_band[::-1]
    result.extend(keltner_upper_band)
    return result


# This calculates the total number of times the bands were crossed for a period of values
def calculate_keltner_intersection(value_list, kbands):
    count = 0

    for i in range(1, len(value_list)):
        # For each value
        for band in kbands:
            # For each band value, check if the line from previous value to current value
            # crosses the keltner band. If yes, increase count
            if value_list[i - 1] < band < value_list[i]:
                count += 1
            elif value_list[i - 1] > band > value_list[i]:
                count += 1

    return count


# Calculate r value using the formula given by professor
def calculate_r(mean_values, i):
    if i > 0:
        # ri = (Pi − Pi−1)/Pi−1
        if mean_values[i - 1] != 0:
            return (mean_values[i] - mean_values[i - 1]) / mean_values[i - 1]
    return 0


# Calculate if we need to continue investing in the currency or not
def calculate_trailing_loss_and_reinvest(
    currency_pair, currency_bought, returns_list, current_return, hour
):
    # Total profit / Total Investment
    profit_percent = (sum(returns_list) * 100) / (currency_bought + sum(returns_list))

    if currency_pair in long_trades:
        return profit_percent > stop_loss_values[hour]
    else:
        return profit_percent < (-1 * stop_loss_values[hour])


# This main function repeatedly calls the polygon api every 1 seconds for 24 hours
# and stores the results.
def main(currency_pairs):
    with open("result.csv", "w", newline="") as csvfile:
        with open("currency.csv", "w", newline="") as currency_file:

            csvwriter = csv.writer(csvfile, delimiter=",")
            csvwriter.writerow(
                ["Reading", "Currency", "Max", "Min", "Mean", "Vol", "FD", "R"]
            )
            csvwriter2 = csv.writer(currency_file, delimiter=",")
            csvwriter2.writerow(["Hour", "Currency", "Bought", "Profit/Loss"])

            # The api key given by the professor
            key = RestClient.fetch_key()

            # Initializing the in-memory data structures
            counter = 0
            six_min_counter = 0
            hour_counter = 0
            keltner_band = {}
            value_list = {}
            r_values = {}
            mean_values = {}
            currency_bought = {}
            profit_values = {}
            r_sum_values = {}
            invest_more = {}
            reset_hour = False

            for currency in currency_pairs:
                name = currency[0] + "_" + currency[1]
                value_list[name] = []
                keltner_band[name] = []
                mean_values[name] = []
                r_values[name] = []
                r_sum_values[name] = []
                profit_values[name] = []
                currency_bought[name] = [100]
                invest_more[name] = True

            # Open a RESTClient for making the api calls
            client = RESTClient(key)

            # Loop that runs until the total duration of the program hits 24 hours.
            while counter <= 18020:  # 18000 seconds = 10 hours
                t0 = time.time()

                # Make a check to see if 6 minutes has been reached or not
                if six_min_counter == 180:
                    for currency in currency_pairs:
                        name = currency[0] + "_" + currency[1]

                        max_val = max(value_list[name])
                        min_val = min(value_list[name])
                        mean_val = round(mean(value_list[name]), 7)
                        mean_values[name].append(mean_val)

                        # Volatility
                        vol = round((max_val - min_val), 7)

                        # If volatility is 0 - divide by 0 error, so skip
                        if vol == 0:
                            fd = "N/A"
                        else:
                            # Get the n number
                            n_number = calculate_keltner_intersection(
                                value_list[name], keltner_band[name]
                            )

                            # Calculate the Fractal Dimension
                            fd = n_number / (vol * 100)

                        r_value = round(
                            calculate_r(mean_values[name], (int(counter / 180) - 1)), 7
                        )
                        r_values[name].append(r_value)

                        csvwriter.writerow(
                            [
                                int(counter / 180),
                                name,
                                max_val,
                                min_val,
                                mean_val,
                                vol,
                                fd,
                                r_value,
                            ]
                        )
                        csvfile.flush()

                        # Calculate the new Keltner Bands
                        keltner_band[name] = get_keltner_channel(mean_val, vol)
                        value_list[name] = []

                        if hour_counter == 1800:
                            r_sum = sum(r_values[name])
                            if invest_more[name]:
                                hour = int(counter / 1800)

                                invest_more[
                                    name
                                ] = calculate_trailing_loss_and_reinvest(
                                    name,
                                    currency_bought[name][-1],
                                    r_sum_values[name],
                                    r_sum,
                                    hour,
                                )

                                if invest_more[name]:
                                    currency_bought[name].append(
                                        currency_bought[name][-1] + 100
                                    )
                                r_sum_values[name].append(r_sum)
                            r_values[name] = []
                            csvwriter2.writerow(
                                [hour, name, currency_bought[name][-1], r_sum]
                            )
                            currency_file.flush()
                            reset_hour = True

                    if reset_hour:
                        reset_hour = False
                        hour_counter = 0
                    six_min_counter = 0

                # Increment the counters
                counter += 1
                six_min_counter += 1
                hour_counter += 1

                # Loop through each currency pair
                for currency in currency_pairs:
                    # Set the input variables to the API
                    from_ = currency[0]
                    to = currency[1]
                    name = from_ + "_" + to

                    # Call the API with the required parameters
                    try:
                        resp = client.get_real_time_currency_conversion(
                            from_, to, amount=100, precision=4
                        )
                    except:
                        continue

                    # This gets the Last Trade object defined in the API Resource
                    last_trade = resp.last

                    # Calculate the price by taking the average of the bid and ask prices
                    avg_price = (last_trade.bid + last_trade.ask) / 2

                    # Append average prices to list
                    value_list[name].append(round(avg_price, 7))

                # Calculate the run time of the function
                t1 = time.time()
                run_time = t1 - t0

                # Only call the api every 1 second, wait for (1-(seconds to execute the previous part))
                # time.sleep((1 - run_time) if run_time < 1 else 0)


# Run the main data collection loop
main(pairs)
