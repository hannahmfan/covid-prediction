import numpy as np
import pandas as pd
import random

file_to_validate = "Final Submission Models/submissions/final_submission.csv"
predictions = pd.read_csv(file_to_validate)

MAX_PRED = 300
LARGE_UPPER_QUANTILE = 10

large_counties = [36061, 17031, 36059, 26163,  6037, 36103, 34013, 34003, 25017, 36119]

medium_counties = [9001, 42101,  9003, 34017, 34039, 26125, 34023,  9009, 34031, 25009,
                   25025, 26099, 25021, 42091, 34029, 25027, 34027, 12086, 18097, 53033]

dates_of_interest = ["2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29", "2020-05-30",
                     "2020-05-31", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05",
                     "2020-06-06", "2020-06-07"]

predictions = predictions[predictions["id"].str.startswith(tuple(dates_of_interest))]


for index, row in predictions.iterrows():
    cols = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
    
    last_val = 0
    for col in cols:
        # Verify that the quantiles are non-decreasing
        if float(row[col]) < last_val:
            print("Error with row_id = %s: quantiles are not increasing" % (row["id"]))

        # Verify that predictions are smaller than MAX_PRED
        if float(row[col]) > MAX_PRED:
            print("Warining, row_id = %s: predicting %f in this row, which seems too high." % (row["id"], row[col]))

        # Verify that all numbers are non negative
        if float(row[col]) < 0:
            print("Error with row_id = %s: negative values" % (row["id"]))

    for county in large_counties:
        if "-" + str(county) + "-" in row["id"]:
            if float(row["90"]) < LARGE_UPPER_QUANTILE:
                print("Warining, row_id = %s: predicting %f for 90 percentile, which seems too low." % (row["id"], row[col]))

    