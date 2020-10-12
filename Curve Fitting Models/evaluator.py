# Author: Jake Will 
#
# A script that allows us to locally evaluate our model's performance

import pandas as pd
import numpy as np

# Requires two NumPy arrays as input, the truth in y_true and predictions in y_pred. 
# The quantile should be a number between 0 and 1. I copied this code from the
# piazza post describing how to compute the pinball loss.
def pinball_loss(y_true, y_pred, quantile = 0.5):
    delta = y_true - y_pred
    # Compute loss for underestimates. 
    loss_above = np.sum(delta[delta > 0]) * (quantile)
    # Compute loss for overestimates.
    loss_below = np.sum(-1 * delta[delta < 0]) * (1 - quantile)
    return (loss_above + loss_below) / len(y_true)

# Input the name of the submission file to evaluate here
submission_file = "linear_fit_submission.csv"

# Input the desired dates into these lists - both lists
# need to be updated because the files have different
# date formats


#nyt_dates = ['4/25/20', '4/26/20', '4/27/20', '4/28/20', '4/29/20', '4/30/20', '5/1/20', '5/2/20', '5/3/20', '5/4/20']
#submission_dates = ['2020-04-25', '2020-04-26', '2020-04-27', '2020-04-28', '2020-04-29', '2020-04-30', '2020-05-01', '2020-05-02', '2020-05-03', '2020-05-04']

submission_dates = ['2020-05-01', '2020-05-02', '2020-05-03', '2020-05-04', '2020-05-05', '2020-05-06', '2020-05-07', '2020-05-08']

# Compute the submission predictions
submission = pd.read_csv(submission_file)
submission = submission[submission['id'].str.contains(('|'.join(submission_dates)))]

# Compute the actual results
deaths = pd.read_csv("../data/us/covid/nyt_us_counties_daily.csv")
deaths = deaths[['date', 'fips', 'deaths']]
deaths = deaths[deaths['date'].str.contains(('|'.join(submission_dates)))]

# Generate a numpy array of the actual results in the same order
# as the submission. If a county has no reported deaths, we assume
# that is has 0.

truth = np.empty(len(submission['id'].values))
for i, submission_id in enumerate(submission['id'].values):
    split_id = submission_id.split('-')
    # Extract the FIPS and date from the id column of the submission
    FIPS = int(split_id[-1])
    date = '-'.join(split_id[:-1])

    # Extract the relevant row of the nyt deaths data
    df = deaths.loc[(deaths['fips'] == FIPS) & (deaths['date'] == date)]

    # Set the truth numpy array accordingly
    if df.empty:
        truth[i] = 0
    else:
        truth[i] = df['deaths']

# Compute the pinball score using the given dates, submission, and
# truth values
score = 0.0
for column in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    score = score + pinball_loss(truth, submission[str(column)].values, quantile = column / 100.0)

score = score/9.0

print(score)