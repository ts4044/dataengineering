## PREDICTING RETURNS USING REGRESSION

This project is a 2 day experiment that is used to predict returns of a currency over time. The currency pairs are :

```
"EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDHKD", "USDAUD", "USDNZD", "USDSGD".
```

## Understanding the process

1) We first collect the data from Polygon website on Day 1 for 10 hours. The script performs the following actions:
    - Gets the raw data from Polygon website and stores in <currency>_raw table in day1.db sqlite database.
    - Every 6 minutes, aggregates the data and calculates mean, min, max values of the currency trading rates.
      Calculates Volatility, Fractal Dimensions and Returns. Also, calculates the Keltner Bands to be used for the next
      6 minutes.
    - volatility = (max value - min value) / mean value
    - fractal dimension = Keltner Band Intersections / Volatility
    - returns = 𝑟𝑖 = (𝑃𝑖 − 𝑃𝑖−1) ⁄ (𝑃𝑖−1) for time period i
    - Every 1 hour, checks if the returns are within a certain treshold and chooses whether to invest in the currency or
      not. (Stop loss)
    - These actions are performed for 10 hours to obtain 100 data points for Volatility, FD, Returns and 10 data points
      for Stop Loss.
    - Generates csv files containing the results for these actions under data folder as <currency>.csv and <currency>_
      output.csv.

2) After Day 1 Data collection is completed, we now need to build prediction models for the return values. We use
   pycaret regression modules to achieve this. Each currency pair has a model build for it using the following
   technique:
    - Classify the Volatility and FD by sorting them and considering the top 33 as high (1), mid 33 (2) as medium and
      low 33 as low (3).
    - Compare every regression technique available and choose the best model and save it.
    - These models are saved under model folder.

3) Finally, we are ready to apply the predictions on real time data. The trained models are used to predict returns on
   real time data.

## Steps to execute the code

1) Run the jupyter notebook - DataCollection.ipynb
2) Run the jupyter notebook - CreateModels.ipynb
3) Run the jupyter notebook - Predict.ipynb