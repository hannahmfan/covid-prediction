import numpy as np
import pandas as pd
import random

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from scipy.optimize import OptimizeWarning

import warnings

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', HessianInversionWarning)
warnings.simplefilter('ignore', OptimizeWarning)
warnings.simplefilter('ignore', RuntimeWarning)

################################################################
############# Util class used throughout the file ##############
################################################################

class util:
    def __init__(self, daily_deaths, cumulative_deaths, key):
        self.daily_deaths       = daily_deaths
        self.cumulative_deaths  = cumulative_deaths
        self.key                = key

    # Get a list of all the deaths by day for a given county,
    # starting from the date of the first case and ending
    # at the given date. Includes both endpoints.
    def get_deaths_list(self, FIPS, endDate):
        if not '-' in endDate:
            sys.exit("Error: endDate in wrong format. Use yyyy-mm-dd. Called from get_deaths_list in util")

        rows = self.daily_deaths.loc[self.daily_deaths["fips"] == FIPS]
        deaths_list = rows["deaths"].values
        dates_list = rows["date"].values

        if endDate in dates_list:
            index = list(dates_list).index(endDate)
        else:
            return []

        return deaths_list[0:index+1]

    # Returns true if there exists deaths data for the county,
    # and false otherwise. We need this because some FIPS are
    # not included in the ny times data at all.
    def deaths_data_exists(self, FIPS):
        return len(self.daily_deaths.loc[self.daily_deaths["fips"] == FIPS].values) != 0

    # Given the key dataframe and a FIPS code, return the name
    # of the county associated with the code.
    def get_name_from_fips(self, FIPS):
        return self.key.loc[self.key["FIPS"] == FIPS]["COUNTY"].values[0]

    





################################################################
############### Global Dataframes and Variables ################
################################################################

# We import the sample_submission.csv file as a way of determining
# the order of the rows in out output file
sample_submission = pd.read_csv("../sample_submission.csv")

# The fips_key.csv file contains standard information about each county
key = pd.read_csv("../data/us/processing_data/fips_key.csv", encoding='latin-1')

# Daily deaths contains the death count per day for each county.
# Cumulative deaths contains the total death count for each county
# by day.
daily_deaths = pd.read_csv("../data/us/covid/nyt_us_counties_daily.csv")
cumulative_deaths = pd.read_csv("../data/us/covid/deaths.csv")

arima_util = util(daily_deaths, cumulative_deaths, key)

# List of all counties
all_fips = key["FIPS"].tolist()

# Relevent dates
today = cumulative_deaths.columns[-1]
yesterday = cumulative_deaths.columns[-2]
one_week_ago = cumulative_deaths.columns[-8]
two_weeks_ago = cumulative_deaths.columns[-15]
beginning = cumulative_deaths.columns[4]







################################################################
################### Global Helper Functions ####################
################################################################

# Assume date is in format mm/dd/yy, convert to yyyy-mm-dd
def convert_date_to_yyyy_mm_dd(date):
    parts = date.split('/')
    
    # Ensure leading zeros if necessary
    if len(parts[0]) == 1:
        parts[0] = "0" + parts[0]
    
    if len(parts[1]) == 1:
        parts[1] = "0" + parts[1]
        
    return "2020" + "-" + parts[0] + "-" + parts[1]

# Assume date is in format yyyy-mm-dd, convert to mm/dd/yy
def convert_date_to_mm_dd_yy(date):
    parts = date.split('-')
    
    # Remove leading zeros if necessary
    if parts[1][0] == "0":
        parts[1] = parts[1][1:]
    
    if parts[2][0] == "0":
        parts[2] = parts[2][1:]
        
    return parts[1] + "/" + parts[2] + "/" + "20"

# Get the name of a county from a given FIPS code
def get_name_from_fips(FIPS):
    return key.loc[key["FIPS"] == FIPS]["COUNTY"].values[0]








################################################################
############## Functions used to Train the Model ###############
################################################################

def train_arima(trainData, order=(2, 1, 0)):
    model = ARIMA(trainData, order=order)
    model_fit = model.fit(disp=0)

    return model_fit


def get_death_predictions_for_county(FIPS, numSteps, startDate):
    # If there is not data on the number of deaths, just return a
    # list of zeros
    if not arima_util.deaths_data_exists(FIPS):
        return [0] * numSteps
    
    deaths = arima_util.get_deaths_list(FIPS, endDate=startDate)

    if len(deaths) == 0:
        return [0] * numSteps
    
    # If the arima model fails, return the most recent number of
    # deaths that is known (often times this is due to a relatively
    # constant time series, so predicting the most recent value is
    # not necessarily a bad choice).
    try:
        model = train_arima(deaths, order=(2, 1, 0))
    except:
        return [deaths[-1]] * numSteps
    
    
    forecast = model.forecast(steps=numSteps)[0]
    
    return list(forecast)








################################################################
######################### Run the Model ########################
################################################################

class ArimaModel:
    def get_predictions_for_all_counties(self, last_train_date, n_pred_steps):
        predictions = {}

        for fips in all_fips:
            preds = get_death_predictions_for_county(fips, n_pred_steps, last_train_date)
            predictions[fips] = preds

        return predictions

    def get_predictions_for_county(self, fips, last_train_date, n_pred_steps):
        return get_death_predictions_for_county(fips, n_pred_steps, last_train_date)
