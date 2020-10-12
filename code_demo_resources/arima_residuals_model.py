from scipy.special import erf
from scipy.optimize import curve_fit

import numpy as np
from numpy import array

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from scipy.optimize import OptimizeWarning

import warnings

import pandas as pd

from datetime import datetime


warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', HessianInversionWarning)
warnings.simplefilter('ignore', OptimizeWarning)
warnings.simplefilter('ignore', RuntimeWarning)

################################################################
############# Util class used throughout the file ##############
################################################################

class util:
    def __init__(self, daily_deaths, cumulative_deaths, county_land_areas, county_populations, mobility_data, key):
        self.daily_deaths       = daily_deaths
        self.cumulative_deaths  = cumulative_deaths
        self.county_land_areas  = county_land_areas
        self.county_populations = county_populations
        self.mobility_data      = mobility_data
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

    def get_total_population(self, FIPS):
        county = self.county_populations.loc[self.county_populations["FIPS"] == FIPS]

        if county.empty:
            sys.exit("Error: no population data. Called from get_total_population in imperial_model.py.")

        return county["total_pop"].values[0]

    def get_mobility_data(self, FIPS):
        county_name = self.get_name_from_fips(FIPS)
        county_state = self.get_state_from_fips(FIPS)

        if FIPS == 11001:
            df = self.mobility_data.loc[(self.mobility_data["sub_region_1"] == "District of Columbia")]
        elif FIPS == 17043:
            df = self.mobility_data.loc[(self.mobility_data["sub_region_1"] == "Illinois") & \
                                        (self.mobility_data["sub_region_2"] == "DuPage County")]
        elif FIPS == 24510:
            df = self.mobility_data.loc[(self.mobility_data["sub_region_1"] == "Maryland") & \
                                        (self.mobility_data["sub_region_2"] == "Baltimore")]
        else:
            df = self.mobility_data.loc[(self.mobility_data["sub_region_1"] == county_state) & \
                                        (self.mobility_data["sub_region_2"] == county_name)]

        if df.empty:
            print("FIPS: " + str(FIPS));
            print("County Name: " + self.get_name_from_fips(FIPS))
            sys.exit("Error: no mobility data. Called from get_mobility_data in util.")

        retail_and_recreation = df["retail_and_recreation_percent_change_from_baseline"].values
        #grocery_and_pharmacy  = df["grocery_and_pharmacy_percent_change_from_baseline"].values[:-4]
        #parks                 = df["parks_percent_change_from_baseline"].values
        transit_stations      = df["transit_stations_percent_change_from_baseline"].values
        workplaces            = df["workplaces_percent_change_from_baseline"].values
        residential           = df["residential_percent_change_from_baseline"].values

        return [retail_and_recreation, transit_stations, workplaces, residential]

    # Given the key dataframe and a FIPS code, return the name
    # of the county associated with the code.
    def get_name_from_fips(self, FIPS):
        return self.key.loc[self.key["FIPS"] == FIPS]["COUNTY"].values[0]

    # Given the key dataframe and a FIPS code, return the state
    # of the county associated with the code.
    def get_state_from_fips(self, FIPS):
        state_abbreviations = pd.read_csv("data/us/processing_data/state_abbreviations.csv")
        abbreviations = state_abbreviations["Abbreviation"].values
        states = state_abbreviations["State"].values

        state_map = {}
        for i in range(len(abbreviations)):
            state_map[abbreviations[i]] = states[i]

        return state_map[self.key.loc[self.key["FIPS"] == FIPS]["ST"].values[0]]







################################################################
############### Global Dataframes and Variables ################
################################################################


# We import the sample_submission.csv file as a way of determining
# the order of the rows in out output file
sample_submission = pd.read_csv("sample_submission.csv")

# The fips_key.csv file contains standard information about each county
key = pd.read_csv("data/us/processing_data/fips_key.csv", encoding='latin-1')

# Daily deaths contains the death count per day for each county.
# Cumulative deaths contains the total death count for each county
# by day.
daily_deaths = pd.read_csv("data/us/covid/nyt_us_counties_daily.csv")
cumulative_deaths = pd.read_csv("data/us/covid/deaths.csv")
county_land_areas = pd.read_csv("data/us/demographics/county_land_areas.csv", encoding='latin1')
county_population = pd.read_csv("data/us/demographics/county_populations.csv", encoding='latin1')
mobility_data = pd.read_csv("data/us/mobility/DL-us-m50.csv", encoding='latin1')

# List of all counties
all_fips = key["FIPS"].tolist()

MIN_TOTAL_DEATHS = 80
MIN_DAYS_SINCE_FIRST_DEATH = 10

today = cumulative_deaths.columns[-1]

arima_util = util(daily_deaths, cumulative_deaths, county_land_areas, county_population, mobility_data, key)






################################################################
#################### Helpful Date Functions ####################
################################################################

# Get all dates used over the course of the term
all_dates = sample_submission["id"].values.copy()

def extract_date_from_id(row_id):
    split = row_id.split('-')
    return '-'.join(split[:-1])

for i in range(len(all_dates)):
    all_dates[i] = extract_date_from_id(all_dates[i])

# Remove duplicates in the list
all_dates = list(dict.fromkeys(all_dates))

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

# Starting from a given date, take an input number of steps
# and compute a list of dates containg the start date and
# "steps" dates into the future or past, for a total of steps
# dates.
def get_dates_from_start(startDate, steps):
    if steps > 0:
        dates = all_dates[all_dates.index(startDate):all_dates.index(startDate) + steps]
    else:
        dates = all_dates[all_dates.index(startDate) + steps:all_dates.index(startDate)]
    return dates

# Get the next date of a given date
def get_next_date(startDate):
    return get_dates_from_start(startDate, 2)[1]







################################################################
################### Curve Fitting Functions ####################
################################################################

def erf_curve(times, log_max, slope, center):
    max_val = 10 ** log_max
    deaths = max_val * (1 + erf(slope * (times - center)))
    return deaths

def eval_erf(times, coefs):
    max_val = 10 ** coefs[0]
    deaths = max_val * (1 + erf(coefs[1] * (times - coefs[2])))
    return deaths

def linear_curve(times, slope, intercept):
    return [x * slope for x in times] + intercept

def constant_curve(times, c):
    return [x * c for x in times]






################################################################
################### Miscellaneous Functions ####################
################################################################

# Given a list of daily deaths, compute a list of cumulative
# deaths of the same length
def get_cumulative_from_daily(daily):
    cumulative = []
    curr = 0
    for deaths in daily:
        curr += deaths
        cumulative.append(curr)
    
    return cumulative

# Given a fips and end date, fit an erf curve to the cumulative deaths
# of the county and return the coefficients
def get_erf_curve(fips, endDate):
    daily_deaths_list = arima_util.get_deaths_list(fips, endDate=endDate)
    cumulative_deaths_list = get_cumulative_from_daily(daily_deaths_list)

    # Compute x and y lists to pass to curve_fit
    x = [i for i in range(len(cumulative_deaths_list))]
    y = cumulative_deaths_list
    
    assert len(y) >= MIN_DAYS_SINCE_FIRST_DEATH and y[-1] > MIN_TOTAL_DEATHS
    popt, pcov = curve_fit(erf_curve, x, y, maxfev=10000)
    
    return popt

def get_erf_residuals(fips, end_train_date, n_steps):
    train_daily_deaths = arima_util.get_deaths_list(fips, endDate=end_train_date)
    all_daily_deaths = arima_util.get_deaths_list(fips, endDate=convert_date_to_yyyy_mm_dd(today))
    
    # Get an optimal erf curve fit for this county up to end_train_date
    erf_coefs = get_erf_curve(fips, end_train_date)
    
    # Ensure that there are n_steps more dates after the end of the train date
    assert len(train_daily_deaths) + n_steps <= len(all_daily_deaths)
    
    # Generate an input array to evaluate predictions on the coming n_steps dates
    x_input = []
    for i in range(len(train_daily_deaths), len(train_daily_deaths) + n_steps):
        x_input.append(i)
    
    cumulative_train_deaths = get_cumulative_from_daily(train_daily_deaths)
    all_cumulative_deaths = get_cumulative_from_daily(all_daily_deaths)
    
    # Make predictions for the next n_steps days
    predictions = []
    for i in x_input:
        predictions.append(eval_erf(i, erf_coefs))
    
    # Compute the residuals of the predictions
    residuals = []
    for i, pred in enumerate(predictions):
        residuals.append(all_cumulative_deaths[x_input[i]] - pred)
        
    assert len(residuals) == n_steps
    
    return residuals, erf_coefs

def get_id_list():
    return sample_submission["id"].values

def extract_date_from_id(row_id):
    split = row_id.split('-')
    return '-'.join(split[:-1])

def extract_fips_from_id(row_id):
    return row_id.split('-')[-1]

def train_arima(trainData, order=(2, 1, 0)):
    model = ARIMA(trainData, order=order)
    model_fit = model.fit(disp=0)

    return model_fit

def get_residuals_predictions(train_residuals, n_steps, order=(2, 1, 0)):
    try:
        model = train_arima(train_residuals, order=order)
    except:
        average = np.mean(train_residuals)
        return [average] * n_steps
    
    forecast = model.forecast(steps=n_steps)[0]
    return list(forecast)






################################################################
######################### Run the Model ########################
################################################################

class ArimaResidualsModel:
    def get_predictions_for_all_counties(self, last_train_date, n_pred_steps):
        dates_to_consider = get_dates_from_start(get_next_date(last_train_date), n_pred_steps)

        data = {}
        for n_erf_pred_steps in range(7, 16):
            residuals_map = {}
            erf_coefs_map = {}

            x_train = []
            y_train = []

            for fips in all_fips:
                last_erf_train_date = get_dates_from_start(dates_to_consider[0], -(n_erf_pred_steps + 1))[0]

                daily_deaths_list = arima_util.get_deaths_list(fips, endDate=last_erf_train_date)
                cumulative_deaths_list = get_cumulative_from_daily(daily_deaths_list)

                if len(cumulative_deaths_list) <= MIN_DAYS_SINCE_FIRST_DEATH or cumulative_deaths_list[-1] <= MIN_TOTAL_DEATHS:
                    continue

                residuals, erf_coefs = get_erf_residuals(fips, last_erf_train_date, n_erf_pred_steps)

                residuals_map[fips] = residuals
                erf_coefs_map[fips] = erf_coefs

            # Store predictions in a dictionary

            for fips in all_fips:
                if fips in [44001, 44003, 44005, 44007, 44009]:
                    data[fips] = [0] * n_pred_steps
                    continue

                daily_deaths_list = arima_util.get_deaths_list(fips, endDate=last_train_date)
                cumulative_deaths_list = get_cumulative_from_daily(daily_deaths_list)

                if (len(cumulative_deaths_list) == 0) or (not fips in residuals_map and cumulative_deaths_list[-1] < 20):
                    data[fips] = [0] * n_pred_steps
                elif not fips in residuals_map:
                    # Fit a linear model to the last 20 points of data
                    length = min(20, len(daily_deaths_list))
                    x_input = [i for i in range(length)]

                    popt, pcov = curve_fit(linear_curve, x_input, daily_deaths_list[-length:], maxfev=10000)

                    x_preds = [i + length for i in range(n_pred_steps)]
                    output = linear_curve(x_preds, popt[0], popt[1])

                    if fips in data:
                        data[fips] = [data[fips][i] + output[i] / 9 for i in range(n_pred_steps)]
                    else:
                        data[fips] = list([x / 9 for x in output])
                else:
                    residuals = residuals_map[fips]
                    erf_coefs = erf_coefs_map[fips]

                    daily_deaths = arima_util.get_deaths_list(fips, endDate=last_train_date)
                    cumulative_deaths = get_cumulative_from_daily(daily_deaths)

                    residuals_predictions = get_residuals_predictions(residuals, n_pred_steps)
                    x_in = [i + len(cumulative_deaths) for i in range(0, n_pred_steps)]
                    erf_predictions = eval_erf(x_in, erf_coefs)

                    final_predictions = list(residuals_predictions + erf_predictions)
                    final_predictions = list(np.diff(final_predictions))
                    final_predictions.insert(0, final_predictions[0])

                    if fips in data:
                        data[fips] = [data[fips][i] + final_predictions[i] / 9 for i in range(n_pred_steps)]
                    else:
                        data[fips] = list([x / 9 for x in final_predictions])

        return data