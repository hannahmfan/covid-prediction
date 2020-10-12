import pystan
import pandas as pd
import numpy as np
import pandas as pd

import datetime
from datetime import date

from heapq import nlargest

import os
os.environ['STAN_NUM_THREADS'] = "4"





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





################################################################
################### STAN code for the model ####################
################################################################

model = """
data {
    int<lower=1> M;                    // Number of counties
    int<lower=1> N;                 // Days of observed data
    int<lower=1> predict_days;         // Number of days to predict
    
    int reported_cases[M, N];
    int reported_deaths[M, N];
    
    real pi[N];
}

parameters {
    real<lower=0> death_rate[M];
    real<lower=0> phi[M];
}

transformed parameters {
    matrix[M, N + predict_days] E_deaths;
    
    for (m in 1:M) {
        for (i in 1:19) {
            E_deaths[m, i] = 1e-9;
        }
    }
    
    for (m in 1:M) {
        for (i in 20:N) {
            E_deaths[m, i] = 1e-9;
            for (j in 1:(i-1)) {
                E_deaths[m, i] += reported_cases[m, j] * pi[i - j] * death_rate[m];
            }
        }
    }
    
    for (m in 1:M) {
        for (i in (N+1):(N+predict_days)) {
            E_deaths[m, i] = 1e-9;
            for (j in 20:N) {
                E_deaths[m, i] += reported_cases[m, j] * pi[i - j] * death_rate[m];
            }
        }
    }
}

model {
    for (m in 1:M) {
        death_rate[m] ~ normal(0.1, 0.05);
        phi[m] ~ normal(0,5);
    }
    
    for (m in 1:M) {
        for (i in 20:N) {
            reported_deaths[m, i] ~ neg_binomial_2(E_deaths[m, i], phi[m]);
        }
    }   
}
"""




################################################################
############### Global Dataframes and Variables ################
################################################################


# We import the sample_submission.csv file as a way of determining
# the order of the rows in out output file
sample_submission = pd.read_csv("sample_submission.csv")

# Dataframes containing deaths, cases, demographic, and mobility data
key = pd.read_csv("data/us/processing_data/fips_key.csv", encoding='latin-1', low_memory=False)
daily_deaths = pd.read_csv("data/us/covid/nyt_us_counties_daily.csv")
cumulative_deaths = pd.read_csv("data/us/covid/deaths.csv")
county_land_areas = pd.read_csv("data/us/demographics/county_land_areas.csv", encoding='latin1', low_memory=False)
county_population = pd.read_csv("data/us/demographics/county_populations.csv", encoding='latin1', low_memory=False)
mobility_data = pd.read_csv("data/google_mobility/Global_Mobility_Report.csv", encoding='latin1', low_memory=False)

age_data = pd.read_csv("data/us/advanced_age_data/PEP_2018_PEPAGESEX_with_ann.csv", encoding='latin1', low_memory=False)

# List of all counties
all_fips = key["FIPS"].tolist()

utils = util(daily_deaths, cumulative_deaths, county_land_areas, county_population, mobility_data, key)

df = pd.read_csv("code_demo_resources/distributions/AdjustedPi.csv")
pi_values = list(df.values.reshape(1, 200)[0])



################################################################
####################### Global Functions #######################
################################################################

def get_top_cumuluative_death_counties(count, endDate):
    all_fips = key["FIPS"].tolist()
    cumulative_deaths_map = {}
    for fips in all_fips:
        deaths = utils.get_deaths_list(fips, endDate=endDate)
        cumulative_deaths_map[fips] = np.sum(deaths)
    
    n_largest = nlargest(count, cumulative_deaths_map, key = cumulative_deaths_map.get)
    return n_largest

def construct_data_for_pystan(end_date, fips_list, num_pred_days, num_past_days):
    data = {}
    data["M"] = len(fips_list)
    data["N"] = num_past_days
    data["predict_days"] = num_pred_days
    data["pi"] = pi_values[:data["N"]]
    
    reported_cases = []
    reported_deaths = []
    
    for fips in fips_list:
        cases = utils.get_cases_list(fips, end_date)[-num_past_days:]
        deaths = utils.get_deaths_list(fips, end_date)[-num_past_days:]
        
        reported_cases.append([int(x) for x in cases])
        reported_deaths.append([int(x) for x in deaths])
    
    data["reported_cases"] = reported_cases
    data["reported_deaths"] = reported_deaths
    
    return data






################################################################
######################### Run the Model ########################
################################################################

class CaseDistributionModel:
    def __init__(self):
        self.data = None
        self.num_pred_days = -1
        self.fips_list = []

    def train(self, fips_list, end_train_date, num_pred_days, num_past_days, iters, chains):
        print("Compiling model...")

        extra_compile_args = ['-pthread', '-DSTAN_THREADS']
        cases_model = pystan.StanModel(model_code=model, extra_compile_args=extra_compile_args)

        print("Training model...")

        data = construct_data_for_pystan(end_train_date, fips_list, num_pred_days, num_past_days)

        control = {'max_treedepth': 20, 'adapt_delta': 0.8}
        fit = cases_model.sampling(data=data, iter=iters, chains=chains, warmup=int(iters/2), thin=4, control=control, n_jobs=8)

        print("Training compelted.")

        self.data = data
        self.num_pred_days = num_pred_days
        self.fips_list = fips_list

        return fit

    def get_predictions_map(self, fit):
        assert(self.data != None)
        assert(self.num_pred_days > 0)
        assert(len(self.fips_list) > 0)

        summary_dict = fit.summary(probs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        df = pd.DataFrame(summary_dict['summary'], 
                          columns=summary_dict['summary_colnames'], 
                          index=summary_dict['summary_rownames'])

        predictions = {}
        for index, fips in enumerate(self.fips_list):
            preds_list = []
            N = self.data['N']

            for i in range(N+1, N + self.num_pred_days + 1):
                preds_list.append(df.loc[df.index == "E_deaths[" + str(index + 1) + "," + str(i) + "]"].values[0][7])
        
            predictions[fips] = preds_list

        return predictions
