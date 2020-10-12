# A custom python package that provides useful features when
# training the Imperial College model for Covid-19 spread.
#
# Author: Jake Will

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime
from datetime import date

import sys

class util:
    def __init__(self, daily_deaths, cumulative_deaths, county_land_areas, county_populations, mobility_data, key):
        self.daily_deaths       = daily_deaths
        self.cumulative_deaths  = cumulative_deaths
        self.county_land_areas  = county_land_areas
        self.county_populations = county_populations
        self.mobility_data      = mobility_data
        self.key                = key

    # Assume date is in format mm/dd/yy, convert to yyyy-mm-dd
    def convert_date_to_yyyy_mm_dd(self, date):
        parts = date.split('/')
        
        # Ensure leading zeros if necessary
        if len(parts[0]) == 1:
            parts[0] = "0" + parts[0]
        
        if len(parts[1]) == 1:
            parts[1] = "0" + parts[1]
            
        return "2020" + "-" + parts[0] + "-" + parts[1]

    # Assume date is in format yyyy-mm-dd, convert to mm/dd/yy
    def convert_date_to_mm_dd_yy(self, date):
        parts = date.split('-')
        
        # Remove leading zeros if necessary
        if parts[1][0] == "0":
            parts[1] = parts[1][1:]
        
        if parts[2][0] == "0":
            parts[2] = parts[2][1:]
            
        return parts[1] + "/" + parts[2] + "/" + "20"

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

    def get_population_per_square_mile(self, FIPS):
        county = self.county_land_areas.loc[self.county_land_areas["County FIPS"] == FIPS]

        if county.empty:
            return 20
            #sys.exit("Error: no population data. Called from get_population_per_square_mile in util.")

        return county["2010 Density per square mile of land area - Population"].values[0]

    def get_square_miles(self, FIPS):
        county = self.county_land_areas.loc[self.county_land_areas["County FIPS"] == FIPS]

        if county.empty:
            return 1000

        return county["Area in square miles - Total area"].values[0]

    def get_total_population(self, FIPS):
        county = self.county_populations.loc[self.county_populations["FIPS"] == FIPS]

        if county.empty:
            return 20
            #sys.exit("Error: no population data. Called from get_population_per_square_mile in util.")

        return county["total_pop"].values[0]

    def get_percentage_over_60(self, FIPS):
        total_pop = self.get_total_population(FIPS)

        county = self.county_populations.loc[self.county_populations["FIPS"] == FIPS]

        if county.empty:
            return 15
            #sys.exit("Error: no population data. Called from get_population_per_square_mile in util.")

        return county["60plus"].values[0] / total_pop

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
        state_abbreviations = pd.read_csv("../data/us/processing_data/state_abbreviations.csv")
        abbreviations = state_abbreviations["Abbreviation"].values
        states = state_abbreviations["State"].values

        state_map = {}
        for i in range(len(abbreviations)):
            state_map[abbreviations[i]] = states[i]

        return state_map[self.key.loc[self.key["FIPS"] == FIPS]["ST"].values[0]]

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

    def plot_list(self, data):
        plt.plot(data, color='red')
        plt.show()