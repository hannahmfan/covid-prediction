import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd

import datetime
from datetime import date

from scipy.stats import norm
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
        state_abbreviations = pd.read_csv("../data/us/processing_data/state_abbreviations.csv")
        abbreviations = state_abbreviations["Abbreviation"].values
        states = state_abbreviations["State"].values

        state_map = {}
        for i in range(len(abbreviations)):
            state_map[abbreviations[i]] = states[i]

        return state_map[self.key.loc[self.key["FIPS"] == FIPS]["ST"].values[0]]





################################################################
################### STAN code for the model ####################
################################################################

model = """
data {
    int<lower=1> M;          // Number of counties
    int<lower=1> N0;         // Number of days for which to impute infections.
    int<lower=1> N[M];       // Days of observed data.
    int<lower=1> N2;         // Days of observed data + # of days to forecast.
    int deaths[M, N2];       // Reported deaths. Ignore all i > N[m].
    real pi[M, N2];          // Infection to death distribution. Pre-calculated and passed as data.
    real SI[N2];             // Pre-calculated serial interval.
    real mobility[M, 4, N2]; // Mobility data
    int full_lockdown[M, N2]; // Denotes when full lockdowns were implemented across the country
    real damping_factor[M, N2];
    
    int populations[M];       // County populations
    int EpidemicStart[M];    // Start of the epidemic for each county
}

parameters {
    real<lower=0> m[M];     // Mean for mu
    real<lower=0> mu[M];    // Intercept for R_t
    real<lower=0> kappa;
    real<lower=0> alpha[M, 4]; // Coefficients for mobility data
    real<lower=0> beta[M];     // Used for full lockdown
    real<lower=0> y[M];
    real<lower=0> phi;
    real<lower=0> tau[M];
}

transformed parameters {    
    matrix[M, N2] predicted_cases    = rep_matrix(0, M, N2);
    matrix[M, N2] E_deaths           = rep_matrix(0, M, N2);
    real<lower=0> R_t[M, N2];
    
    /* Fill in the initial cases for each county */
    for (i in 1:M) {
        for (j in 1:N0) {
            predicted_cases[i, j] = y[i];
        }
    }
    
    /* Compute the R_t coefficients for each county */
    for (i in 1:M) {
        for (j in 1:N2) {
            /* Compute the R_t value at this point in time for this county. */
            R_t[i, j] = mu[i] * exp(alpha[i, 1] * mobility[i, 1, j] +
                                    alpha[i, 2] * mobility[i, 2, j] + alpha[i, 3] * mobility[i, 3, j] +
                                    alpha[i, 4] * mobility[i, 4, j] -
                                    beta[i] * full_lockdown[i][j]) * damping_factor[i][j];
        }
    }
    
    for (i in 1:M) {
        for (j in (N0+1):N2) {
            vector[j-1] case_summands;
            for (k in 1:(j-1)) {
                case_summands[k] = predicted_cases[i, k] * SI[j - k];
            }

            predicted_cases[i, j] = R_t[i, j] * sum(case_summands);
        }
    }

    /* Compute the expected deaths for each county */
    for (i in 1:M) {
        E_deaths[i, 1] = 1e-9;
        for (j in 2:N2) {
            vector[j-1] summands;
            for (k in 1:(j - 1)) {
                summands[k] = predicted_cases[i, k] * pi[i, j - k];
            }
            
            E_deaths[i, j] = sum(summands);
        }
    }
}

model {
    phi ~ normal(0,2);
    kappa ~ normal(0, 0.5);
        
    for (i in 1:M) {
        tau[i] ~ exponential(0.03);
        y[i] ~ exponential(1/tau[i]);

        m[i] ~ normal(2.4, 0.6);
        mu[i] ~ normal(m[i], kappa);
        
        for (j in 1:4) {
            alpha[i, j] ~ normal(2, 1);
        }
        beta[i] ~ gamma(0.5, 1);
    }
    
    for (i in 1:M) {
        for (j in EpidemicStart[i]:N[i]) {
            deaths[i, j] ~ neg_binomial_2(E_deaths[i, j], phi);
        }
    }
}

generated quantities {
    matrix[M, N2] predicted_cases0    = rep_matrix(0, M, N2);
    matrix[M, N2] E_deaths0           = rep_matrix(0, M, N2);
    matrix[M, N2] R_t0                = rep_matrix(0, M, N2);
    
    /* Fill in the initial cases for each county */
    for (i in 1:M) {
        for (j in 1:N0) {
            predicted_cases0[i, j] = y[i];
        }
    }
    
    /* Compute the R_t coefficients for each county */
    for (i in 1:M) {
        for (j in 1:N2) {
            /* Compute the R_t value at this point in time for this county. */
            R_t0[i, j] = mu[i] * exp(alpha[i, 1] * mobility[i, 1, j] +
                                     alpha[i, 2] * mobility[i, 2, j] + alpha[i, 3] * mobility[i, 3, j] +
                                     alpha[i, 4] * mobility[i, 4, j] -
                                     beta[i] * full_lockdown[i][j]) * damping_factor[i, j];
        }
    }
    
    for (i in 1:M) {
        for (j in (N0+1):N2) {
            vector[j-1] case_summands;
            for (k in 1:(j-1)) {
                case_summands[k] = predicted_cases[i, k] * SI[j - k];
            }

            predicted_cases0[i, j] = R_t0[i, j] * sum(case_summands);
        }
    }

    /* Compute the expected deaths for each county */
    for (i in 1:M) {
        E_deaths0[i, 1] = 1e-9;
        for (j in 2:N2) {
            vector[j-1] summands;
            for (k in 1:(j - 1)) {
                summands[k] = predicted_cases0[i, k] * pi[i, j - k];
            }
            
            E_deaths0[i, j] = sum(summands);
        }
    }
}
"""





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

imperial_util = util(daily_deaths, cumulative_deaths, county_land_areas, county_population, mobility_data, key)

# Relevant distributions
df = pd.read_csv("PiDistribution.csv")
pi_values = list(df.values.reshape(1, 200)[0])

df = pd.read_csv("SIDistribution.csv")
SI_values = list(df.values.reshape(1, 200)[0])


################################################################
####################### Global Functions #######################
################################################################

def get_days_between_dates(date1, date2):
    first_date  = datetime.datetime.strptime(date1, "%Y-%m-%d").date()
    second_date = datetime.datetime.strptime(date2, "%Y-%m-%d").date()
    
    return (second_date - first_date).days

def get_n_days_from_date(date, N):
    return_date = datetime.datetime.strptime(date, "%Y-%m-%d").date() + datetime.timedelta(days=N)
    return return_date.strftime('%Y-%m-%d')

def get_county_death_rate(FIPS):
    column_weight_map = {
        "est72017sex0_age0to4": 0,
        "est72017sex0_age5to9": 0,
        "est72017sex0_age10to14": 0.2,
        "est72017sex0_age15to19": 0.2,
        "est72017sex0_age20to24": 0.2,
        "est72017sex0_age25to29": 0.2,
        "est72017sex0_age30to34": 0.2,
        "est72017sex0_age35to39": 0.2,
        "est72017sex0_age40to44": 0.4,
        "est72017sex0_age45to49": 0.4,
        "est72017sex0_age50to54": 1.3,
        "est72017sex0_age55to59": 1.3,
        "est72017sex0_age60to64": 3.6,
        "est72017sex0_age65to69": 3.6,
        "est72017sex0_age70to74": 8.0,
        "est72017sex0_age75to79": 8.0,
        "est72017sex0_age80to84": 14.8,
        "est72018sex2_age85plus": 14.8
    }

    total_pop = 0
    for name in column_weight_map:
        total_pop += int(age_data.loc[age_data["GEO.id2"] == FIPS][name].values[0])

    death_rate = 0
    for name in column_weight_map:
        pop = int(age_data.loc[age_data["GEO.id2"] == FIPS][name].values[0])
        death_rate += (pop / total_pop) * column_weight_map[name]

    return death_rate

def get_top_cumuluative_death_counties(count, endDate):
    all_fips = key["FIPS"].tolist()
    cumulative_deaths_map = {}
    for fips in all_fips:
        deaths = imperial_util.get_deaths_list(fips, endDate=endDate)
        cumulative_deaths_map[fips] = np.sum(deaths)
    
    n_largest = nlargest(count, cumulative_deaths_map, key = cumulative_deaths_map.get)
    return n_largest

# Smooth mobility list to remove large upwards spikes
def smooth_mobility_list(mobility):
    for i in range(3, len(mobility)):
        window_average = np.mean(mobility[i-3:i])
        
        # Remove very large upwards spikes
        if (mobility[i] - window_average) > 20:
            mobility[i] = window_average

# Smooth deaths list using a moving average
def smooth_deaths_list(deaths):
    for i in range(3, len(deaths) - 3):
        window_average = np.mean(deaths[i-3:i+4])
        deaths[i] = window_average


# Takes in a given end date and num_counties parameters
# and outputs a dictionary that can be passed in to fit
# the imperial college model using PyStan.
#
# num_counties specifies the top num_counties, ordered
# by top daily deaths.
def construct_data_for_pystan(end_date, all_fips, num_pred_days, last_mobility_date):
    first_date = "2020-02-15"
    N = get_days_between_dates(first_date, end_date) + 1
    
    all_deaths_lists       = []
    epidemic_start_indices = []
    mobility_lists         = []
    populations            = []
    full_lockdown          = []
    damping_factor         = []
    pi_distributions       = []
    
    # Compute the deaths list for each county
    for fips in all_fips:
        deaths_list = list(imperial_util.get_deaths_list(fips, endDate=end_date))
        deaths_list = deaths_list[-N:]
        
        # Ensure the list has length at least N
        if len(deaths_list) < N:
            deaths_list = [0] * (N - len(deaths_list)) + deaths_list
        
        smooth_deaths_list(deaths_list)
            
        # Impute -1 for days to be predicted
        deaths_list = deaths_list + [-1] * num_pred_days
        
        deaths_list = [int(x) for x in deaths_list]
        all_deaths_lists.append(deaths_list)
    
    # Compute the start of the epidemic for each county
    for deaths_list in all_deaths_lists:
        for i in range(len(deaths_list)):
            if deaths_list[i] != 0:
                epidemic_start_indices.append(max(1, i - 20))
                break
                    
    # We conveniently choose the start date to be 2/15, so just truncate the end if needed
    for i, fips in enumerate(all_fips):
        mobility = imperial_util.get_mobility_data(fips)

        for j in range(len(mobility)):
            if len(mobility[0]) > N:
                mobility[j] = mobility[j][0:N]
            
            average = np.mean(mobility[j][-30:])
            
            impute_days = get_days_between_dates(last_mobility_date, end_date)

            mobility[j] = list(mobility[j]) + [average] * impute_days
            mobility[j] = list(mobility[j]) + [average] * num_pred_days
            
            mobility[j] = [x / 100 for x in mobility[j]]
        
        # Convert residential values to be negative
        mobility[3] = [-x for x in mobility[3]]
                
        # Impute data for any NaN
        for j in range(len(mobility)):
            for k in range(len(mobility[j])):
                if str(mobility[j][k]) == "nan":
                    if k == 0:
                        mobility[j][k] = 0.1
                    elif k < 10:
                        mobility[j][k] = np.mean(mobility[j][0:k])
                    else:
                        mobility[j][k] = np.mean(mobility[j][(k-10):k])

        # Handle large outliers
        for j in range(len(mobility)):
            smooth_mobility_list(mobility[j])

        mobility_lists.append(mobility)
    
    for i, fips in enumerate(all_fips):
        population = imperial_util.get_total_population(fips)
        populations.append(population)
        
    for i, fips in enumerate(all_fips):
        if fips == 36061:
            index = get_days_between_dates(first_date, "2020-03-20")
        else:
            index = get_days_between_dates(first_date, "2020-03-25")
            
        full_lockdown.append([0] * index + [1] * (len(all_deaths_lists[0]) - index))
        
        damping = []
        for j in range(len(all_deaths_lists[i])):
            if full_lockdown[i][j] == 0:
                damping.append(1)
            else:
                damping.append(damping[j - 1] * 0.99)
                
        damping_factor.append(damping)
    
    N2 = get_days_between_dates(first_date, end_date) + 1 + num_pred_days
    
    for fips in all_fips:
        pi = pi_values[:N2]
        try:
            death_rate = get_county_death_rate(str(fips))
        except:
            death_rate = 0.02
        
        pi_distributions.append([x * death_rate for x in pi])
    
    data = {}
    data['M']             = len(all_fips)
    data['N0']            = 3
    data['N']             = [get_days_between_dates(first_date, end_date) + 1] * len(all_fips)
    data['N2']            = get_days_between_dates(first_date, end_date) + 1 + num_pred_days
    data['deaths']        = all_deaths_lists
    data['pi']            = pi_distributions
    data['SI']            = SI_values[:data['N2']]
    data['mobility']      = mobility_lists
    data['EpidemicStart'] = epidemic_start_indices
    data['populations']   = populations
    data['full_lockdown'] = full_lockdown
    data['damping_factor'] = damping_factor
    
    return data, all_fips


################################################################
######################### Run the Model ########################
################################################################

class ImperialModel:
    def __init__(self):
        self.data = None
        self.num_pred_days = -1
        self.fips_list = []

    def train_imperial_model(self, fips_list, end_train_date, n_pred_days, end_mobility_date, iters, chains):
        print("Compiling model...")

        extra_compile_args = ['-pthread', '-DSTAN_THREADS']
        imperial_model = pystan.StanModel(model_code=model, extra_compile_args=extra_compile_args)

        print("Training model...")

        data, fips_list = construct_data_for_pystan(end_train_date, fips_list, 14, end_mobility_date)
        num_pred_days = n_pred_days

        control = {'max_treedepth': 20, 'adapt_delta': 0.8}
        fit = imperial_model.sampling(data=data, iter=iters, chains=chains, warmup=int(iters/2), thin=4, control=control, n_jobs=8)

        print("Training completed.")

        self.data = data
        self.num_pred_days = num_pred_days
        self.fips_list = fips_list

        return fit

    def get_imperial_predictions_map(self, fit):
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
            N = self.data['N2']

            for i in range(N - self.num_pred_days, N + 1):
                preds_list.append(df.loc[df.index == "E_deaths0[" + str(index + 1) + "," + str(i) + "]"].values[0][7])
        
            predictions[fips] = preds_list

        return predictions