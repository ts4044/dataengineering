## PREDICTIVE STOP LOSS USING REGRESSION

This project is a 2 day experiment that is used to predict if a currency is worth investing in. The currency pairs are :
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

3) Finally, we are ready to apply the predictions on real time data. The trained models are used to obtain errors using
   which we will update our stop loss function to consider the following:
    - If the estimated and actual values for the previous hour are signal ALIGNED (i.e., if both are
      simultaneously long or simultaneously short), and the error is SMALL, then we REINVEST.
    - If the estimated and actual values for the previous hour are signal ALIGNED, and the error is
      NOT SMALL, then we DO NOTHING.
    - If the estimated and actual values for the previous hour are signal DIVERGENT (i.e., if one is
      pointing long and the other is pointing short), then we STOP the position, whatever the error is.
    - All of the results are presented as csv files in the folder.

## Steps to execute the code
1) Run the jupyter notebook - DataCollection.ipynb under 1datacollection folder.
2) Run the jupyter notebook - CreateModels.ipynb under 2createmodels folder.
3) Run the jupyter notebook - Day2.ipynb under 3predictions folder.