import numpy as np
from scipy.integrate import odeint
import pandas as pd
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp
import math, datetime

################################################################
####################### Helper Methods #########################
################################################################

def get_derivatives(y, t, N, gamma, alpha, r_d, r_i, r_ri, r_rh, r_dth, p_dth, p_d, p_h):
    S, E, I, AR, AD, DHR, DHD, DQR, DQD, R, D = y
    
    dSdt = -alpha * gamma * S * I / N
    dEdt = alpha * gamma * S * I / N - r_i * E
    dIdt = r_i * E - r_d * I
    dARdt = r_d * (1 - p_dth) * (1 - p_d) * I - r_ri * AR
    dDHRdt = r_d * (1 - p_dth) * p_d * p_h * I - r_rh * DHR
    dDQRdt = r_d * (1 - p_dth) * p_d * (1 - p_h) * I - r_ri * DQR
    dADdt = r_d * p_dth * (1 - p_d) * I - r_dth * AD
    dDHDdt = r_d * p_dth * p_d * p_h * I - r_dth * DHD
    dDQDdt = r_d * p_dth * p_d * (1 - p_h) * I - r_dth * DQD
    dRdt = r_ri * (AR + DQR) + r_rh * DHR
    dDdt = r_dth * (AD + DQD + DHD)
    
    return dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt, dDQDdt, dRdt, dDdt

# Compute RMSE of daily cases, daily deaths, cumulative cases, and
# cumulative deaths, and return a weighted sum
def test_fit(S, E, I, AR, AD, DHR, DHD, DQR, DQD, R, D, df, 
    case_wt=0, death_wt=2, c_case_wt=0, c_death_wt=0.05):
    case_err, death_err, c_case_err, c_death_err = 0, 0, 0, 0
    
    for i, row in df.iterrows():
        if i == 0:
            this_cases = DHR[i] + DHD[i] + DQR[i] + DQD[i]
            this_deaths = D[i]
        else:
            this_cases = DHR[i] + DHD[i] + DQR[i] + DQD[i] - (DHR[i-1] + DHD[i-1] + DQR[i-1] + DQD[i-1])
            this_deaths = D[i] - D[i-1]
    
        case_err += (row["cases"] - this_cases) ** 2
        death_err += (row["deaths"] - this_deaths) ** 2
        c_case_err += (row["c_cases"] - (DHR[i] + DHD[i] + DQR[i] + DQD[i])) ** 2
        c_death_err += (row["c_deaths"] - D[i]) ** 2
    
    case_err /= len(S)
    death_err /= len(S)
    c_case_err /= len(S)
    c_death_err /= len(S)
    
    return (case_wt * math.sqrt(case_err) + death_wt * math.sqrt(death_err) + 
            c_case_wt * math.sqrt(c_case_err) + c_death_wt * math.sqrt(c_death_err))

################################################################
###################### Parameter Tuning ########################
################################################################

class HyperOpt(object):
    def __init__(self, population, data, y_init, timespace):
        self.data = data.copy()
        self.data.reset_index(inplace=True)
        self.y_init = y_init
        self.pop = population
        self.t = timespace
    
    def eval_model(self, params):
        result = odeint(get_derivatives, self.y_init, self.t, args=(self.pop, 
                                                                    params["gamma"],
                                                                    params["alpha"], 
                                                                    math.log(2) / params["T_d"], 
                                                                    math.log(2) / params["T_i"], 
                                                                    math.log(2) / params["T_ri"], 
                                                                    math.log(2) / params["T_rh"],
                                                                    math.log(2) / params["T_dth"],
                                                                    params["p_dth"], params["p_d"], params["p_h"]))
        
        S, E, I, AR, AD, DHR, DHD, DQR, DQD, R, D = result.T
        rmse = test_fit(S, E, I, AR, AD, DHR, DHD, DQR, DQD, R, D, self.data)
        return rmse
    
    def optimize_params(self, space, trials, algo, max_evals):
        result = fmin(fn=self.eval_model, space=space, algo=algo, max_evals=max_evals, trials=trials, verbose=False)
        return result, trials

################################################################
############################ Data ##############################
################################################################

data = pd.read_csv("data/us/covid/nyt_us_counties_daily.csv")
population = pd.read_csv("data/us/demographics/county_populations.csv")
fips_list = pd.read_csv("data/us/processing_data/fips_key.csv", encoding="cp1252")
sample = pd.read_csv("sample_submission.csv")

################################################################
###################### Model Parameters ########################
################################################################

# Assume these parameters are constant across all counties
T_d = 2                                 # Days to detection
T_i = 5                                 # Days to leave incubation
T_ri = 10                               # Days to recovery not in hospital
T_rh = 15                               # Days to recovery in hospital
gamma = 1                               # Don't model government response yet
p_d = 0.2                               # Percentage of cases detected
p_h = 0.15                              # Percentage of detected cases hospitalized

# For generating a normal distribution
z_80 = 1.28
z_60 = 0.84
z_40 = 0.525
z_20 = 0.25

################################################################
####################### Model Training #########################
################################################################

def get_cumulative(df):
    df["c_deaths"], df["c_cases"] = 0, 0
    for i, row in df.iterrows():   
        try:
            df.at[i, "c_deaths"] = df.loc[i - 1, "c_deaths"] + df.loc[i, "deaths"]
            df.at[i, "c_cases"] = df.loc[i - 1, "c_cases"] + df.loc[i, "cases"]
        except Exception as e:
            df.at[i, "c_deaths"] = df.loc[i, "deaths"]
            df.at[i, "c_cases"] = df.loc[i, "cases"]
            
    return df

def fit_delphi_params(df, infection_start, pop, y_init):
    t = np.linspace(0, len(df)-infection_start, len(df)-infection_start)

    param_space = {
        "gamma": gamma,
        "alpha": hp.uniform("alpha", 0.2, 0.4),
        "T_d": T_d, "T_i": T_i, "T_ri": T_ri, "T_rh": T_rh,
        "T_dth": hp.uniform("T_dth", 5, 20),
        "p_dth": hp.uniform("p_dth", 0.01, 0.06),
        "p_d": p_d, "p_h": p_h
    }

    hopt = HyperOpt(pop, df[infection_start:], y_init, t)
    optimized, trials = hopt.optimize_params(space=param_space, trials=Trials(), algo=tpe.suggest, max_evals=100)
    
    return optimized    


################################################################
#################### Generate Predictions ######################
################################################################

class DelphiModel:
    
    # use_sample, submission, drop_negative are only used by predict_all_counties. Default values are for using this method by itself
    def predict_one_county(county, last_train_date, num_pred_steps, submission=None, use_sample=True, drop_negative=True, verbose=True):
        if verbose: print("County " + str(county) + "...", end='\r', flush=True)
            
        if use_sample: submission = sample.copy()
            
        county_data = data.loc[data["fips"] == county]
        last_date = datetime.datetime.fromisoformat(data.iloc[-1]["date"])
        train_date = datetime.datetime.fromisoformat(last_train_date)
        test_per = (last_date - train_date).days

        if test_per > 0: df = county_data[:-test_per]
        else: df = county_data
        df.reset_index(inplace=True)
        df = get_cumulative(df)

        try: 
            cum_deaths = df.iloc[-1]["c_deaths"]
        except IndexError as e:
            if len(df) == 0:
                if verbose: print("No data found for county", str(county))
                return submission
            else:
                cum_deaths = 0

        if cum_deaths >= 15:
            try:
                pop = int(population.loc[population["FIPS"] == county]["total_pop"])
            except TypeError as e:
                print("No population found for county", str(county))
                print("This county has at least 15 cumulative deaths!")
                raise e

            # Find first cases
            infection_start = df.loc[df["cases"] > 0].first_valid_index()
            start_date = df.iloc[infection_start]["date"]

            # Initial numbers
            # Assume initial detected cases are evenly distributed between death/recovery and quarantine/hospital
            E_init, AR_init, AD_init, R_init = 0, 0, 0, 0
            I_init, D_init = df.iloc[infection_start]["cases"], df.iloc[infection_start]["deaths"]
            DHR_init, DHD_init, DQR_init, DQD_init = I_init / 4, I_init / 4, I_init / 4, I_init / 4
            S_init = pop - I_init - D_init

            y_init = S_init, E_init, I_init, AR_init, AD_init, DHR_init, DHD_init, DQR_init, DQD_init, R_init, D_init
            
            optimized = fit_delphi_params(df, infection_start, pop, y_init)
            
            t = np.linspace(0, len(df) + num_pred_steps, len(df) + num_pred_steps)
            res = odeint(get_derivatives, y_init, t, args=(pop, gamma, optimized["alpha"], 
                                                           math.log(2) / T_d, math.log(2) / T_i, math.log(2) / T_ri, 
                                                           math.log(2) / T_rh, math.log(2) / optimized["T_dth"],
                                                           optimized["p_dth"], p_d, p_h))

            S, E, I, AR, AD, DHR, DHD, DQR, DQD, R, D = res.T

            max_deaths = 0.000005 * pop

            date = datetime.date.fromisoformat(df.iloc[0]["date"]) + datetime.timedelta(days=int(infection_start))
            for i, ddata in enumerate(D):
                this_id = date.isoformat() + "-" + str(county)
                date += datetime.timedelta(days=1)

                if i == 0: mid = ddata
                else: mid = ddata - D[i - 1]

                if mid > max_deaths: mid = max_deaths

                sd = 3 * math.sqrt(mid)

                try:
                    ss_location = submission.index[submission["id"] == str(this_id)][0]
                    submission.at[ss_location,"10"] = mid - sd * z_80
                    submission.at[ss_location,"20"] = mid - sd * z_60
                    submission.at[ss_location,"30"] = mid - sd * z_40
                    submission.at[ss_location,"40"] = mid - sd * z_20
                    submission.at[ss_location,"50"] = mid
                    submission.at[ss_location,"60"] = mid + sd * z_20
                    submission.at[ss_location,"70"] = mid + sd * z_40
                    submission.at[ss_location,"80"] = mid + sd * z_60
                    submission.at[ss_location,"90"] = mid + sd * z_80
                except IndexError as e:
                    #print(date)
                    continue
        
        if drop_negative:
            submission["10"] = submission["10"].apply(lambda x: x if x >= 1 else 0)
            submission["20"] = submission["20"].apply(lambda x: x if x >= 1 else 0)
            submission["30"] = submission["30"].apply(lambda x: x if x >= 1 else 0)
            submission["40"] = submission["40"].apply(lambda x: x if x >= 1 else 0)
            submission["50"] = submission["50"].apply(lambda x: x if x >= 1 else 0)
            submission["60"] = submission["60"].apply(lambda x: x if x >= 1 else 0)
            submission["70"] = submission["70"].apply(lambda x: x if x >= 1 else 0)
            submission["80"] = submission["80"].apply(lambda x: x if x >= 1 else 0)
            submission["90"] = submission["90"].apply(lambda x: x if x >= 1 else 0)
        
        return submission
    
    def predict_all_counties(last_train_date, num_pred_steps, verbose=True):
        if verbose: print("Started at: " + str(datetime.datetime.now()) + "\n")
            
        submission = sample.copy()
        for idx, row in fips_list.iterrows():
            submission = DelphiModel.predict_one_county(
                int(row["FIPS"]), last_train_date, num_pred_steps, submission=submission, use_sample=False, drop_negative=False, verbose=verbose
            )
        
        submission["10"] = submission["10"].apply(lambda x: x if x >= 1 else 0)
        submission["20"] = submission["20"].apply(lambda x: x if x >= 1 else 0)
        submission["30"] = submission["30"].apply(lambda x: x if x >= 1 else 0)
        submission["40"] = submission["40"].apply(lambda x: x if x >= 1 else 0)
        submission["50"] = submission["50"].apply(lambda x: x if x >= 1 else 0)
        submission["60"] = submission["60"].apply(lambda x: x if x >= 1 else 0)
        submission["70"] = submission["70"].apply(lambda x: x if x >= 1 else 0)
        submission["80"] = submission["80"].apply(lambda x: x if x >= 1 else 0)
        submission["90"] = submission["90"].apply(lambda x: x if x >= 1 else 0)
        
        if verbose: print("\nFinished at: " + str(datetime.datetime.now()))
            
        return submission
    