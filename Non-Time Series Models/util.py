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

class util:
    def __init__(self, daily_deaths, cumulative_deaths, county_land_areas, county_populations, education, beds, key):
        self.daily_deaths       = daily_deaths
        self.cumulative_deaths  = cumulative_deaths
        self.county_land_areas  = county_land_areas
        self.county_populations = county_populations
        self.education          = education
        self.key                = key
        self.beds               = beds

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
            sys.exit("Error: endDate in wrong format. Use yyyy-mm-dd. Called from get_cases_list in lstm_util")

        rows = self.daily_deaths.loc[self.daily_deaths["fips"] == FIPS]
        cases_list = rows["cases"].values
        dates_list = rows["date"].values

        if endDate in dates_list:
            index = list(dates_list).index(endDate)
        else:
            return []

        return cases_list[0:index+1]

    def get_dates_list(self, FIPS, endDate):
        if not '-' in endDate:
            sys.exit("Error: endDate in wrong format. Use yyyy-mm-dd. Called from get_dates_list in lstm_util")

        rows = self.daily_deaths.loc[self.daily_deaths["fips"] == FIPS]
        dates = rows["date"].values

        if endDate in dates:
            index = list(dates).index(endDate)
        else:
            return []

        return dates[0:index+1]

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

    def get_weekday(self, input_date):
        my_date = datetime.strptime(input_date, "%Y-%m-%d")
        return my_date.weekday()

    def get_population_per_square_mile(self, FIPS):
        county = self.county_land_areas.loc[self.county_land_areas["County FIPS"] == FIPS]

        if county.empty:
            return 20
            #sys.exit("Error: no population data. Called from get_population_per_square_mile in rf_util.")

        return county["2010 Density per square mile of land area - Population"].values[0]

    def get_total_population(self, FIPS):
        county = self.county_populations.loc[self.county_populations["FIPS"] == FIPS]

        if county.empty:
            return 40000

        return county["total_pop"].values[0]

    def get_pop_over_60(self, FIPS):
        county = self.county_populations.loc[self.county_populations["FIPS"] == FIPS]

        if county.empty:
            return 5000

        return county["60plus"].values[0]

    def get_education_data(self, FIPS):
        county = self.education.loc[self.education["FIPS"] == FIPS]

        if county.empty:
            return [20, 40, 20, 20]

        less_than_high_school = county["Percent of adults with less than a high school diploma, 2014-18"].values[0]
        high_school = county["Percent of adults with a high school diploma only, 2014-18"].values[0]
        some_college = county["Percent of adults completing some college or associate's degree, 2014-18"].values[0]
        college = county["Percent of adults with a bachelor's degree or higher, 2014-18"].values[0]
        
        if str(less_than_high_school) == "nan":
            less_than_high_school = 20
        if str(high_school) == "nan":
            high_school = 40
        if str(some_college) == "nan":
            some_college = 20
        if str(college) == "nan":
            college = 20

        return [less_than_high_school, high_school, some_college, college]

    def get_hospital_data(self, FIPS):
        county = self.beds.loc[self.education["FIPS"] == FIPS]

        if county.empty:
            return [100, 10, 1, 0.1]

        staffed_beds = county["staffed_beds"].values[0]
        icu_beds     = county["icu_beds"].values[0]

        if str(staffed_beds) == "nan" or str(icu_beds) == "nan":
            return [100, 10, 1, 0.1]

        population = self.get_total_population(FIPS)

        beds_per_mille = float(staffed_beds) / (population / 1000)
        icu_beds_per_mille = float(icu_beds) / (population / 1000)

        return [staffed_beds, icu_beds, beds_per_mille, icu_beds_per_mille]

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

    def generate_input_data(self, FIPS, date, cases_window_size=14, cases_lag_time=21):
        if not '-' in date:
            sys.exit("Error: date in wrong format. Use yyyy-mm-dd. Called from generate_input_data in lstm_util")

        rows = self.daily_deaths.loc[self.daily_deaths["fips"] == FIPS]
        cases_list = rows["cases"].values
        dates_list = rows["date"].values

        try:
            index = list(dates_list).index(date)
        except:
            index = -1

        # Ensure the cases list is long enough
        if index < cases_lag_time:
            cases_list = [0] * (cases_lag_time - index - 1) + list(cases_list)
            index = cases_lag_time - 1

        X = list(cases_list[index - cases_lag_time + 1: index - cases_lag_time + 1 + cases_window_size])
        #X.append(self.get_population_per_square_mile(FIPS))
        #X.append(self.get_weekday(date))

        #total_pop = self.get_total_population(FIPS)
        #pop_over_60 = self.get_pop_over_60(FIPS)

        #X.append(total_pop)
        #X.append(pop_over_60)
        #X.append(float(pop_over_60)/total_pop)

        #X += list(self.get_education_data(FIPS))
        #X += list(self.get_hospital_data(FIPS))

        return X