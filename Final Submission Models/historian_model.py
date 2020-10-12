import pandas as pd
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

import datetime
from datetime import date

from heapq import nlargest

import math




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

    def get_cases_list(self, FIPS, endDate):
        if not '-' in endDate:
            sys.exit("Error: endDate in wrong format. Use yyyy-mm-dd. Called from get_deaths_list in util")

        rows = self.daily_deaths.loc[self.daily_deaths["fips"] == FIPS]
        cases_list = rows["cases"].values
        dates_list = rows["date"].values

        if endDate in dates_list:
            index = list(dates_list).index(endDate)
        else:
            return []

        return cases_list[0:index+1]
        
    # Given the key dataframe and a FIPS code, return the name
    # of the county associated with the code.
    def get_name_from_fips(self, FIPS):
        return self.key.loc[self.key["FIPS"] == FIPS]["COUNTY"].values[0]

    # Given the key dataframe and a FIPS code, return the state
    # of the county associated with the code.
    def get_state_from_fips(self, FIPS):
        state_abbreviations = pd.read_csv("../data/us/processing_data/state_abbreviations.csv")
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
sample_submission = pd.read_csv("../sample_submission.csv")

# Dataframes containing deaths, cases, demographic, and mobility data
key = pd.read_csv("../data/us/processing_data/fips_key.csv", encoding='latin-1', low_memory=False)
daily_deaths = pd.read_csv("../data/us/covid/nyt_us_counties_daily.csv")
cumulative_deaths = pd.read_csv("../data/us/covid/deaths.csv")
county_land_areas = pd.read_csv("../data/us/demographics/county_land_areas.csv", encoding='latin1', low_memory=False)
county_population = pd.read_csv("../data/us/demographics/county_populations.csv", encoding='latin1', low_memory=False)
mobility_data = pd.read_csv("../data/google_mobility/Global_Mobility_Report.csv", encoding='latin1', low_memory=False)

age_data = pd.read_csv("../data/us/advanced_age_data/PEP_2018_PEPAGESEX_with_ann.csv", encoding='latin1', low_memory=False)

# List of all counties
all_fips = key["FIPS"].tolist()

historian_util = util(daily_deaths, cumulative_deaths, county_land_areas, county_population, mobility_data, key)






################################################################
####################### Global Functions #######################
################################################################

def get_top_cumuluative_death_counties(count, endDate):
    all_fips = key["FIPS"].tolist()
    cumulative_deaths_map = {}
    for fips in all_fips:
        deaths = historian_util.get_deaths_list(fips, endDate=endDate)
        cumulative_deaths_map[fips] = np.sum(deaths)
    
    n_largest = nlargest(count, cumulative_deaths_map, key = cumulative_deaths_map.get)
    return n_largest

def round_down(num, divisor):
    return math.floor(num / divisor) * divisor

def round_up(num, divisor):
    return math.ceil(num / divisor) * divisor

def smooth_list(lst):
    for i in range(0, 3):
        lst[i] = np.mean(lst[i:i+5])
        
    for i in range(3, len(lst) - 3):
        window_average = np.mean(lst[i-3:i+4])
        lst[i] = window_average
        
    for i in range(len(lst) - 3, len(lst)):
        lst[i] = np.mean(lst[i-7:i])



################################################################
######################### Run the Model ########################
################################################################

class HistorianModel:
    def __init__(self, boundary_date):
        self.boundary_date = boundary_date
        self.counties = get_top_cumuluative_death_counties(1000, boundary_date)

    def get_predictions(self, FIPS, num_pred_steps):
        x_in = []
        y_in = []

        index = self.counties.index(FIPS)
        
        lower = round_down(index, 10)
        upper = round_up(index, 10)
        
        if lower == upper:
            upper += 10
        
        predictions = [0] * num_pred_steps
        counts = [0] * num_pred_steps
        
        offsets = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        for offset in offsets:
            for fips in self.counties[lower:upper]:        
                x_in = []
                y_in = []

                deaths = historian_util.get_deaths_list(fips, endDate=self.boundary_date)
                smooth_list(deaths)
                cases = historian_util.get_cases_list(fips, endDate=self.boundary_date)
                smooth_list(cases)

                deaths_ratios = []
                for i in range(len(deaths) - offset * 2, len(deaths) - offset - 1):
                    if deaths[i + offset] == 0:
                        deaths_ratios.append(3)
                    else:
                        deaths_ratios.append(deaths[i] / deaths[i + offset])

                cases_ratios = []
                for i in range(len(cases) - offset * 3, len(cases) - offset * 2 - 1):
                    if cases[i + offset] == 0:
                        cases_ratios.append(3)
                    else:
                        cases_ratios.append(cases[i] / cases[i + 14])

                x_in += list(np.array(cases_ratios).reshape(-1, 1))
                y_in += list(np.array(deaths_ratios).reshape(-1, 1))    

            kernel = DotProduct() + WhiteKernel()
            gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x_in, y_in)

            fips = FIPS

            deaths = historian_util.get_deaths_list(fips, endDate=self.boundary_date)
            smooth_list(deaths)
            cases = historian_util.get_cases_list(fips, endDate=self.boundary_date)
            smooth_list(cases)

            for i in range(len(deaths) - offset, len(deaths) - max(offset - num_pred_steps, 0)):
                if cases[i - offset * 2] == 0:
                    case_ratio = 3
                else:
                    case_ratio = cases[i - offset * 2] / cases[i - offset]
                
                predictions[i - len(deaths) + offset] += deaths[i] * gpr.predict([[case_ratio]])[0]
                counts[i - len(deaths) + offset] += 1
        
        predictions = [predictions[i] / counts[i] for i in range(len(predictions))]
        return predictions