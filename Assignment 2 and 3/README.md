
# Polygon Rest Client

### Assignment 1
This library is a helper to fetch the key for Polygon Client. The library can be imported as follows:
```
from polygon_client import RestClient
```

To fetch the key, you can use the fetch_key() method.
```
key = RestClient.fetch_key()
client = RESTClient(key)
```

### Assignment 2
Run _code.py_. Creates a CSV file of the following format :
```
"Reading Number", "Currency", "Max", "Min", "Mean", "Vol", "FD"
```

### Assignment 3

Run _code.py_. Creates 2 CSV file of the following format :

result.csv
```
"Reading Number", "Currency", "Max", "Min", "Mean", "Vol", "FD", "r"
```

currency.csv
```
"Currency", "Bought", "Profit/Loss"
```

This adds on to the previous assignment with an _r_ value and a new csv file with the currency bought and profit loss values for every hour.