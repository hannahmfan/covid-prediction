import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.svm import SVR
import math

################################################################
############### Global Dataframes and Variables ################
################################################################

# We import the sample_submission.csv file as a way of determining
# the order of the rows in out output file
sample_submission = pd.read_csv("sample_submission.csv")
daily_deaths = pd.read_csv("data/us/covid/nyt_us_counties_daily.csv")
cumulative_deaths = pd.read_csv("data/us/covid/deaths.csv")

# Relevant dates
today = cumulative_deaths.columns[-1]
yesterday = cumulative_deaths.columns[-2]
one_week_ago = cumulative_deaths.columns[-8]
two_weeks_ago = cumulative_deaths.columns[-15]
beginning = cumulative_deaths.columns[4]

print("Today: " + today)
print("One week ago: " + one_week_ago)


################################################################
################### Global Helper Functions ####################
################################################################

def pinball_loss(y_true, y_pred, quantile = 0.5):
    delta = y_true - y_pred
    # Compute loss for underestimates.
    loss_above = np.sum(delta[delta > 0]) * (quantile)
    # Compute loss for overestimates.
    loss_below = np.sum(-1 * delta[delta < 0]) * (1 - quantile)
    return (loss_above + loss_below) / len(y_true)

# Assume date is in format mm/dd/yy, convert to yyyy-mm-dd
def convert_date_to_yyyy_mm_dd(date):
    parts = date.split('/')

    # Ensure leading zeros if necessary
    if len(parts[0]) == 1:
        parts[0] = "0" + parts[0]

    if len(parts[1]) == 1:
        parts[1] = "0" + parts[1]

    return "2020" + "-" + parts[0] + "-" + parts[1]

# Generate the quantiles for a given value and standard error
# according to a normal distribution.
def generate_quantiles(value, err):
    if err == 0:
        return [value] * 9

    quantiles = []
    for quantile in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        q = norm.ppf(quantile, loc=value, scale=err)
        if q > 300:
            q = 300
        quantiles.append(q)

    return quantiles

# Generate quantiles for a given list of values and errors
def generate_list_quantiles(lst, err_lst):
    quantiles = []
    for i in range(len(lst)):
        quantiles.append(generate_quantiles(lst[i], err_lst[i]))

    return quantiles

def get_id_list():
    return sample_submission["id"].values

def extract_date_from_id(row_id):
    split = row_id.split('-')
    return '-'.join(split[:-1])

def extract_fips_from_id(row_id):
    return row_id.split('-')[-1]

# Get all dates used over the course of the term
all_dates = sample_submission["id"].values.copy()

for i in range(len(all_dates)):
    all_dates[i] = extract_date_from_id(all_dates[i])

# Remove duplicates in the list
all_dates = list(dict.fromkeys(all_dates))

# Starting from a given date, take an input number of steps
# and compute a list of dates containg the start date and
# "steps" dates into the future, for a total of steps
# dates.
def get_dates_from_start(startDate, steps):
    dates = all_dates[all_dates.index(startDate):all_dates.index(startDate) + steps]
    return dates

# Get the next date of a given date
def get_next_date(startDate):
    return get_dates_from_start(startDate, 2)[1]

# Get a list of all the deaths by date for a given county,
# starting from the date of the first case
def get_deaths_list(FIPS, endDate=convert_date_to_yyyy_mm_dd(today)):
    # Extract only the rows for this county in order by date
    rows = daily_deaths.loc[daily_deaths["fips"] == FIPS]
    deaths_list = rows["deaths"].values
    dates_list = rows["date"].values

    if endDate in dates_list:
        index = list(dates_list).index(endDate)
    else:
        return []

    return deaths_list[0:index+1]

################################################################
################### Optimal Hyperparameters ####################
################################################################

kernel = 'rbf'
C = 100
gamma = 0.1
epsilon = 0.1

specific_window = 1
window_limit = 11
s = 14
standard_error = 0.75


################################################################
############## Functions used to Train the Model ###############
################################################################

# For a county, trains up to a date, predicts s more days until the input date, and gets the losses.
def validation_loss(fips, s, window, test_end):
    data = get_deaths_list(fips, test_end)
    X, y = get_xy(data, window)
    newX = [x.astype("int") for x in X]
    X = np.array(newX)
    y = np.array(y)

    svr_rbf = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    model = svr_rbf.fit(X[:len(X)-s], y[:len(y)-s])

    train = data[:len(data)-s]
    last = train[len(train)-window:]
    loss = 0
    all_pred = []
    for i in range(s):
        pred = model.predict([last])
        all_pred.append(pred[0])
        true = y[len(y)-s+i]
        loss += (pred[0] - true)**2
        newlast = np.append(last[1:],pred)
        last = newlast
    return math.sqrt(loss/s), pinball_loss(y[len(y)-s:], all_pred)

# For a county, returns the best window size in range [a,b).
def best_window(fips, a, b, s, day):
    best = a
    try:
        rmse, pinball = validation_loss(fips, s, best, day)
    except:
        print("failed at 1")
        return best
    for w in range(a+1,b):
        try:
            new_rmse, new_pinball = validation_loss(fips, s, w, day)
            if new_pinball < pinball:
                pinball = new_pinball
                best = w
        except:
            print("failed at", w)
            break
    return best

# Prepares an array into sliding windows X and y.
def get_xy(arr, window):
    X, y = [], []
    for i in range(len(arr)-window):
        X.append(arr[i:i+window])
        y.append(arr[i+window])
    return X, y

# For a county, trains up to a date and predicts some days ahead of it. Doesn't get losses.
def forecast(fips, window, date, days):
    daily = get_deaths_list(fips, date)
    X, y = get_xy(daily, window)
    newX = [x.astype("int") for x in X]
    X = np.array(newX)
    y = np.array(y)

    svr_rbf = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    model = svr_rbf.fit(X, y)

    last = daily[len(daily)-window:]
    all_pred = []
    for i in range(days):
        pred = model.predict([last])
        all_pred.append(pred[0])
        newlast = np.append(last[1:],pred)
        last = newlast
    return all_pred

################################################################
######################### Run the Model ########################
################################################################

# start_date: last date trained
# n_steps: number of prediction steps
def SVM_Regression(start_date, n_steps):
    key = pd.read_csv("data/us/processing_data/fips_key.csv", encoding='latin-1')
    all_fips = key["FIPS"].tolist()

    dates_of_interest = get_dates_from_start(get_next_date(start_date), n_steps)

    data = {}
    for fips in all_fips:
        data[fips] = {}
        window = best_window(fips, 1, window_limit, s, start_date)
        print(key.loc[key["FIPS"] == fips]["COUNTY"].values[0])
        print("Window:", window)

        try:
            pred = forecast(fips, window, start_date, n_steps)
            print("Pred:\n", pred)
            error = [x * standard_error for x in pred]
        except:
            pred, error = [0] * n_steps, [0] * n_steps
            print("Pred: failed")

        quantiles = generate_list_quantiles(pred, error)
        print("Quantiles:")
        for q in quantiles:
            print(q)
        print()

        for i, date in enumerate(dates_of_interest):
            data[fips][date] = quantiles[i]

        lists = []

    return data
