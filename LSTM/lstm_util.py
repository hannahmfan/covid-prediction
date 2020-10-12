# A custom python package that provides useful features when
# training an LSTM model on the Covid-19 data.
#
# Author: Jake Will

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime
from datetime import date

import sys

class lstm_util:
    def __init__(self, daily_deaths, cumulative_deaths, county_land_areas, key):
        self.daily_deaths      = daily_deaths
        self.cumulative_deaths = cumulative_deaths
        self.county_land_areas = county_land_areas
        self.key               = key

    # Get a list of all the deaths by day for a given county,
    # starting from the date of the first case and ending
    # at the given date. Includes both endpoints.
    def get_deaths_list(self, FIPS, endDate):
        if not '-' in endDate:
            sys.exit("Error: endDate in wrong format. Use yyyy-mm-dd. Called from get_deaths_list in lstm_util")

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
            sys.exit("Error: endDate in wrong format. Use yyyy-mm-dd. Called from get_deaths_list in lstm_util")

        rows = self.daily_deaths.loc[self.daily_deaths["fips"] == FIPS]
        cases_list = rows["cases"].values
        dates_list = rows["date"].values

        if endDate in dates_list:
            index = list(dates_list).index(endDate)
        else:
            return []

        return cases_list[0:index+1]

    # Get the date of the first death for a given county and
    # the dataframe containing the cumulative death count for
    # all counties. If the county has no deaths, return "N/A"
    def get_date_of_first_death(self, FIPS):
        county = self.cumulative_deaths.loc[self.cumulative_deaths["countyFIPS"] == FIPS]
        deaths_dates = county.drop(columns=['countyFIPS', 'County Name', 'State', 'stateFIPS'])

        if len(deaths_dates.values) == 0:
            return "N/A"

        lst = deaths_dates.values[0]

        for i in range(len(lst)):
            if lst[i] != 0:
                return deaths_dates.columns[i]

        return "N/A"

    # Given the key dataframe and a FIPS code, return the name
    # of the county associated with the code.
    def get_name_from_fips(self, FIPS):
        return self.key.loc[self.key["FIPS"] == FIPS]["COUNTY"].values[0]

    def get_population_per_square_mile(self, FIPS):
        county = self.county_land_areas.loc[self.county_land_areas["County FIPS"] == FIPS]

        if county.empty:
            return 20
            #sys.exit("Error: no population data. Called from get_population_per_square_mile in rf_util.")

        return county["2010 Density per square mile of land area - Population"].values[0]

    # For a given county and date, determine the number of days
    # since the first death
    def days_since_first_death(self, FIPS, date_string):
        eval_date = datetime.strptime(date_string, "%Y-%m-%d")
        first_death_date_string = self.get_date_of_first_death(FIPS)

        if first_death_date_string == "N/A":
            return 0

        date_of_first_death = datetime.strptime(first_death_date_string, "%m/%d/%y")
        delta = eval_date - date_of_first_death
        return delta.days

    def generate_train_data(self, deaths_list, steps_in, steps_out):
        X, y = list(), list()
        for i in range(len(deaths_list)):
            end_ix = i + steps_in
            out_end_ix = end_ix + steps_out

            if out_end_ix > len(deaths_list):
                break

            seq_x, seq_y = deaths_list[i:end_ix], deaths_list[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)

        return X, y

    def plot_list(self, data):
        plt.plot(data, color='red')
        plt.show()

    def get_differenced_list(self, data):
        differenced = list()

        for i in range(1, len(data)):
            differenced.append(data[i] - data[i - 1])

        return differenced

    def inverse_differenced_list(self, start_val, differenced_list):
        inversed = list()

        curr = start_val
        for i in range(len(differenced_list)):
            inversed.append(curr + differenced_list[i])
            curr = inversed[-1]

        return inversed