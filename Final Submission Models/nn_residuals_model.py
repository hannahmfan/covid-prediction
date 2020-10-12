import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.special import erf
from scipy.optimize import curve_fit
import time

from datetime import datetime
from datetime import date
from datetime import timedelta

import sys

################################################################
####################### Helper Methods #########################
################################################################

# Provides methods to process data to train a neural network on residuals from an erf curve
class Helper:
    
    def __init__(self, nyt_data, mobility_data, population_data, density_data):
        self.nyt = nyt_data
        self.mobility = mobility_data
        self.population = population_data
        self.density = density_data
    
    def map_counties_to_deaths(self, counties):
        result = {}
        
        for i, row in counties.iterrows():
            county = int(row["FIPS"])
            
            c_data = self.nyt.loc[self.nyt["fips"] == int(county)]
            cum_deaths = 0
            for idx, row in c_data.iterrows(): cum_deaths += row["deaths"]
                
            result[county] = cum_deaths
        
        return result
    
    def construct_training_data(self, predictions, prediction_dates, training_cutoff, case_lag, size_cutoff, size_dict):
        training_data_s = pd.DataFrame()
        training_data_s["days_past_data"] = 0
        training_data_s["lag_cases"] = 0
        training_data_s["mobility"] = 0
        training_data_s["prop_60plus"] = 0
        training_data_s["pop_density"] = 0
        training_data_s["residual"] = 0
        
        training_data_l = pd.DataFrame()
        training_data_l["days_past_data"] = 0
        training_data_l["lag_cases"] = 0
        training_data_l["mobility"] = 0
        training_data_l["prop_60plus"] = 0
        training_data_l["pop_density"] = 0
        training_data_l["residual"] = 0
        
        for idx, row in predictions.iterrows():        
            fips = int(row["id"][11:])
            date = row["id"][:10]

            if not date in prediction_dates: continue

            #print(row["id"], end='\r', flush=True)

            prediction = float(row["50"])
            try: actual = int(self.nyt.loc[self.nyt["fips"] == fips].loc[self.nyt["date"] == date]["deaths"])
            except TypeError as e: continue

            residual = prediction - actual
            
            if residual > 1000 or residual < -1000:
                print("Warning: large residual found!")
                print(fips, date)
                print(residual)
                print()

            delta_time = (datetime.fromisoformat(date) - training_cutoff).days

            lag_date = datetime.fromisoformat(date) - timedelta(days = case_lag)
            try: lag_cases = int(self.nyt.loc[self.nyt["fips"] == fips].loc[self.nyt["date"] == lag_date.isoformat()[:10]]["cases"])
            except TypeError as e: lag_cases = 0

            try: mobil = int(self.mobility.loc[self.mobility["fips"] == fips].loc[self.mobility["date"] == date]["m50"])
            except TypeError as e: continue

            p_row = self.population.loc[self.population["FIPS"] == fips]
            plus60 = float(p_row["60plus"]) / float(p_row["total_pop"])

            dens = float(self.density.loc[self.density["County FIPS"] == fips]["2010 Density per square mile of land area - Population"])
            
            cumulative_deaths = size_dict[fips]
            
            if cumulative_deaths >= size_cutoff and not fips == 36061:
                training_data_l.loc[len(training_data_l)] = [delta_time, lag_cases, mobil, plus60, dens, residual]
            elif not fips == 36061:
                training_data_s.loc[len(training_data_s)] = [delta_time, lag_cases, mobil, plus60, dens, residual]

        return training_data_l, training_data_s
    
    def construct_input(self, train, pred_dates, train_cutoff, lag_per):
        c = pd.DataFrame()
        c["days_past_data"] = 0
        c["lag_cases"] = 0
        c["mobility"] = 0
        c["prop_60plus"] = 0
        c["pop_density"] = 0
        c["residual"] = 0

        for idx, row in train.iterrows():        
            fips = row["fips"]
            date = row["date"]

            if not date in pred_dates: continue

            prediction = float(row["50"])
            try: actual = int(self.nyt.loc[self.nyt["fips"] == fips].loc[self.nyt["date"] == date]["deaths"])
            except TypeError as e: continue

            residual = prediction - actual

            delta_time = (datetime.fromisoformat(date) - train_cutoff).days

            lag_date = datetime.fromisoformat(date) - timedelta(days = lag_per)
            try: lag_cases = int(self.nyt.loc[self.nyt["fips"] == fips].loc[self.nyt["date"] == lag_date.isoformat()[:10]]["cases"])
            except TypeError as e:
                lag_cases = 0

            try: mobil = int(self.mobility.loc[self.mobility["fips"] == fips].loc[self.mobility["date"] == date]["m50"])
            except TypeError as e:
                continue

            p_row = self.population.loc[self.population["FIPS"] == fips]
            plus60 = float(p_row["60plus"]) / float(p_row["total_pop"])

            dens = float(self.density.loc[self.density["County FIPS"] == fips]["2010 Density per square mile of land area - Population"])

            c.loc[len(c)] = [delta_time, lag_cases, mobil, plus60, dens, residual]
        
        return c

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

    def get_mobility_data_list(self, FIPS, endDate, n_steps):
        if not '-' in endDate:
            sys.exit("Error: endDate in wrong format. Use yyyy-mm-dd. Called from get_mobility_data_list in util")

        county = self.mobility_data.loc[self.mobility_data["fips"] == FIPS]
        mobility_data = county.values[0][5:]
        dates_data = self.mobility_data.columns.values[5:]

        if endDate == "2020-04-20":
            endDate = "2020-04-21"

        if endDate in dates_data:
            index = list(dates_data).index(endDate)
        else:
            return []

        return mobility_data[index - n_steps + 1:index + 1]


    # Given the key dataframe and a FIPS code, return the name
    # of the county associated with the code.
    def get_name_from_fips(self, FIPS):
        return self.key.loc[self.key["FIPS"] == FIPS]["COUNTY"].values[0]

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

################################################################
####################### Model Training #########################
################################################################

class ResidualNN:
    
    def __init__(self, training_data, history_len, target_len, STEP, TRAIN_SPLIT, BATCH_SIZE, BUFFER_SIZE):
        data = training_data.values
        self.mean = data.mean(axis=0)
        self.stddev = data.std(axis=0)
        self.data = (data-self.mean)/self.stddev
                
        self.history = history_len
        self.target = target_len
        self.STEP = STEP
        self.TRAIN_SPLIT = TRAIN_SPLIT
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        
        x_train, y_train = self.multivariate_data(0, TRAIN_SPLIT)
        x_val, y_val = self.multivariate_data(TRAIN_SPLIT, None)
        
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self.val_data = val_data.batch(BATCH_SIZE).repeat()
        
        self.init_model(x_train)
        
    def init_model(self, x_train):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_train.shape[-2:]))
        model.add(tf.keras.layers.LSTM(16, activation='relu'))
        model.add(tf.keras.layers.Dense(self.target))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        
        self.model = model
    
    # From TensorFlow website
    def multivariate_data(self, start_index, end_index):
        data = []
        labels = []

        start_index = start_index + self.history
        if end_index is None:
            end_index = len(self.data) - self.target

        for i in range(start_index, end_index):
            indices = range(i-self.history, i, self.STEP)
            data.append(self.data[indices])
            labels.append(self.data[:, -1][i:i+self.target])

        return np.array(data), np.array(labels)

    def fit(self):
        return self.model.fit(
            self.train_data, epochs=10, steps_per_epoch=200, validation_data=self.val_data, validation_steps=50, verbose=0
        )
        
    def model_input_array(self, input_data):
        data = []
    
        start_index = self.history
        end_index = len(input_data)

        for i in range(start_index, end_index):
            indices = range(i-self.history, i, self.STEP)
            data.append(input_data[indices])

        return np.array(data)
    
    def adjust_erf(self, final_pred_dates, predictions, residuals):
        ids, i10, i20, i30, i40, i50, i60, i70, i80, i90, dates, fipss = [], [], [], [], [], [], [], [], [], [], [], []

        for i, date in enumerate(final_pred_dates):
            erf_row = predictions.loc[predictions["date"] == date]
            dates.append(date)
            fipss.append(int(erf_row["fips"]))
            ids.append(date + "-" + str(int(erf_row["fips"])))

            i10.append(float(erf_row["10"]) - residuals[i])
            i20.append(float(erf_row["20"]) - residuals[i])
            i30.append(float(erf_row["30"]) - residuals[i])
            i40.append(float(erf_row["40"]) - residuals[i])
            i50.append(float(erf_row["50"]) - residuals[i])
            i60.append(float(erf_row["60"]) - residuals[i])
            i70.append(float(erf_row["70"]) - residuals[i])
            i80.append(float(erf_row["80"]) - residuals[i])
            i90.append(float(erf_row["90"]) - residuals[i])

        return pd.DataFrame(data={"id":ids, "10":i10, "20":i20, "30":i30, "40":i40, "50":i50, "60":i60, "70":i70, "80":i80, "90":i90, "date":dates, "fips":fipss})
    
    def predict_one_county(self, fips, train_erf, pred_erf, helper, train_cutoff, lag_per, erf_dates, pred_dates):
        c_pred = pred_erf.loc[pred_erf["fips"] == fips]
        c_train = train_erf.loc[train_erf["fips"] == fips]
        
        c_input = helper.construct_input(train=c_train, pred_dates=erf_dates, train_cutoff=train_cutoff, lag_per=lag_per)
        
        c_data = c_input.values
        c_data = (c_data-self.mean)/self.stddev
        
        x = self.model_input_array(c_data)
        
        try: 
            nn_predictions = self.model.predict(x)
            adjusted_nn_residuals = (nn_predictions[-1] * self.stddev[-1] + self.mean[-1])

            p_df = self.adjust_erf(pred_dates, c_pred, adjusted_nn_residuals)

            return p_df
        except ValueError as e:
            # Not enough data to make a prediction with the neural net, so just use the erf curve
            return c_pred

################################################################
############################ Data ##############################
################################################################

nyt_data = pd.read_csv("../data/us/covid/nyt_us_counties_daily.csv")
mobility = pd.read_csv("../data/us/mobility/DL-us-mobility-daterow.csv")
population = pd.read_csv("../data/us/demographics/county_populations.csv")
density = pd.read_csv("../data/us/demographics/county_land_areas.csv", encoding="cp1252")

sample = pd.read_csv("../sample_submission.csv")
fips_list = pd.read_csv("../data/us/processing_data/fips_key.csv", encoding="cp1252")

cumulative_deaths = pd.read_csv("../data/us/covid/deaths.csv")
county_land_areas = pd.read_csv("../data/us/demographics/county_land_areas.csv", encoding='latin1')
mobility_data = pd.read_csv("../data/us/mobility/DL-us-m50.csv", encoding='latin1')
all_fips = fips_list["FIPS"].tolist()

################################################################
##################### Generate erf Curves ######################
################################################################

def erf_curve(times, log_max, slope, center):
    max_val = 10 ** log_max
    deaths = max_val * (1 + erf(slope * (times - center)))
    return deaths

def make_predictions(fips, startDate, endDate, n_steps, util):
    # Use the daily deaths list to compute a list of the cumulative deaths.
    # This is better than directly accessing cumulative deaths because
    # the NY data is faulty, so we can directly replace the daily deaths
    # much more easily
    
    daily_deaths_list = util.get_deaths_list(fips, endDate=endDate)
    
    cumulative_deaths_list = []

    curr = 0
    for i in range(len(daily_deaths_list) - 1):
        curr += daily_deaths_list[i]
        cumulative_deaths_list.append(curr)
    
    # Compute x and y lists to pass to curve_fit
    x = [i for i in range(len(cumulative_deaths_list))]
    y = cumulative_deaths_list
    
    if len(cumulative_deaths_list) < 20 or y[-1] < 50:
        return [0] * n_steps
    
    x_input = [i + len(cumulative_deaths_list) for i in range(n_steps + 1)]
    popt, pcov = curve_fit(erf_curve, x, y, maxfev=10000)
    output = erf_curve(x_input, popt[0], popt[1], popt[2])
    
    # Difference predictions to get daily deaths
    predictions = np.diff(output)
    
    return predictions

def generate_quantiles(value):
    quantiles = []
    for i in range(-4, 5):
        quantiles.append(value + value * 0.2 * i)

    return quantiles

def get_id_list():
    return sample["id"].values

def extract_fips_from_id(row_id):
    return row_id.split('-')[-1]

def extract_date_from_id(row_id):
    split = row_id.split('-')
    return '-'.join(split[:-1])

def format_erf_predictions(data):
    lists = []
    for row_id in get_id_list():
        date = extract_date_from_id(row_id)
        fips = int(extract_fips_from_id(row_id))

        if not fips in data:
            lst = [row_id] + ["%.2f" % 0.00] * 9
            lists.append(lst)
            continue

        if not date in data[fips]:
            lst = [row_id] + ["%.2f" % 0.00] * 9
            lists.append(lst)
            continue

        quantiles = data[fips][date]
        lst = [row_id]

        for q in quantiles:
            if str(q) == "nan":
                lst.append("%.2f" % 0.00)
            elif q < 0:
                lst.append("%.2f" % 0.00)
            else:
                lst.append("%.2f" % q)

        lists.append(lst)

    df = pd.DataFrame(lists, columns=sample.columns)
    
    return df
        
def get_erf_predictions(dates, last_training_date):
    ut = util(nyt_data, cumulative_deaths, county_land_areas, population, mobility_data, fips_list)

    data = {}
    for fips in all_fips:
        data[fips] = {}
        predictions = make_predictions(fips, "2020-03-30", last_training_date, len(dates), ut)

        for i, date in enumerate(dates):
            data[fips][date] = generate_quantiles(predictions[i])
    
    return format_erf_predictions(data)

################################################################
####################### Model Training #########################
################################################################

def train_neural_nets(erf, erf_pred_dates, erf_train_cutoff, target_len):
    helper = Helper(nyt_data, mobility, population, density)
        
    c_map = helper.map_counties_to_deaths(pd.read_csv("../data/us/processing_data/fips_key.csv", encoding="cp1252"))

    training_data_l, training_data_s = helper.construct_training_data(
        predictions = erf, prediction_dates = erf_pred_dates, 
        training_cutoff = datetime.fromisoformat(erf_train_cutoff), case_lag = 14,
        size_cutoff = 200,
        size_dict = c_map
    )

    large_NN = ResidualNN(
        training_data = training_data_l.sample(frac=1), # shuffle input data
        history_len = 7, target_len = target_len,
        STEP = 1, TRAIN_SPLIT = 650, BATCH_SIZE = 64, BUFFER_SIZE = 10000
    )
    
    small_NN = ResidualNN(
        training_data = training_data_s.sample(frac=1),
        history_len = 10, target_len = target_len,
        STEP = 1, TRAIN_SPLIT = 27000, BATCH_SIZE = 256, BUFFER_SIZE = 10000
    )
    
    large_history = large_NN.fit()
    small_history = small_NN.fit()
    
    return large_NN, small_NN

def augment_prediction_dataframe(df):
    df["fips"] = 0
    df["date"] = ""

    for i, row in df.iterrows():
        fips = int(row["id"][11:])
        date = row["id"][:10]
        df.at[i, "fips"] = fips
        df.at[i, "date"] = date
    
    return df

################################################################
#################### Generate Predictions ######################
################################################################

class NN_Residuals_Model:
    
    # final_pred_dates: dates to predict in the final output
    # erf_pred_dates: Dates to predict for the initial erf, using those predictions for residuals
    # erf_training_cutoff: Last data point used for training the initial erf (generally the day before predictions start)
    # final_training_cutoff: Last data point to use as training data (generally day before final predictions start)
    
    def predict_all_counties(final_pred_dates, erf_pred_dates, erf_training_cutoff, final_training_cutoff, verbose=True):
        if verbose: 
            print("Started at: " + str(datetime.now())+"\n")
            print("Generating erf predictions...")
            
        erf_init = get_erf_predictions(erf_pred_dates, erf_training_cutoff)
        erf_init = augment_prediction_dataframe(erf_init)
        erf_final = get_erf_predictions(final_pred_dates, final_training_cutoff)
        erf_final = augment_prediction_dataframe(erf_final)
        
        if verbose: print("Training neural nets...")
        large_NN, small_NN = train_neural_nets(erf_init, erf_pred_dates, erf_training_cutoff)
                
        if verbose: print("Generating predictions...\n")
        submission = sample.copy()
        
        for idx, row in fips_list.iterrows():
            county = int(row["FIPS"])
            print("County " + str(county) + "...", end='\r', flush=True)
            c_row = nyt_data.loc[nyt_data["fips"] == county]

            # Construct the input dataframe
            cum_deaths = 0
            for i, item in c_row.iterrows():
                cum_deaths += int(item["deaths"])

            if cum_deaths > 200:
                pred = large_NN.predict_one_county(
                    county, erf, erf_final, helper, datetime.fromisoformat(erf_training_cutoff), 
                    7, erf_pred_dates, final_pred_dates
                )
            else:
                pred = small_NN.predict_one_county(
                    county, erf, erf_final, helper, datetime.fromisoformat(erf_training_cutoff),
                    7, erf_pred_dates, final_pred_dates
                )

            for i, row in pred.iterrows():
                ss_location = submission.index[submission["id"] == row["id"]][0]
                submission.at[ss_location] = row.drop(["date", "fips"])
        
        submission["10"] = submission["10"].apply(lambda x: x if x >= 1 else 0)
        submission["20"] = submission["20"].apply(lambda x: x if x >= 1 else 0)
        submission["30"] = submission["30"].apply(lambda x: x if x >= 1 else 0)
        submission["40"] = submission["40"].apply(lambda x: x if x >= 1 else 0)
        submission["50"] = submission["50"].apply(lambda x: x if x >= 1 else 0)
        submission["60"] = submission["60"].apply(lambda x: x if x >= 1 else 0)
        submission["70"] = submission["70"].apply(lambda x: x if x >= 1 else 0)
        submission["80"] = submission["80"].apply(lambda x: x if x >= 1 else 0)
        submission["90"] = submission["90"].apply(lambda x: x if x >= 1 else 0)

        if verbose: print("\nFinished at: " + str(datetime.now()))
            
        return submission
   
    def predict_one_county(county, final_pred_dates, erf_pred_dates, erf_training_cutoff, final_training_cutoff):
        erf_init = get_erf_predictions(erf_pred_dates, erf_training_cutoff)
        erf_init = augment_prediction_dataframe(erf_init)
        erf_final = get_erf_predictions(final_pred_dates, final_training_cutoff)
        erf_final = augment_prediction_dataframe(erf_final)
        
        large_NN, small_NN = train_neural_nets(erf_init, erf_pred_dates, erf_training_cutoff)
        
        c_row = nyt_data.loc[nyt_data["fips"] == county]

        # Construct the input dataframe
        cum_deaths = 0
        for i, item in c_row.iterrows():
            cum_deaths += int(item["deaths"])

        if cum_deaths > 200:
            pred = large_NN.predict_one_county(
                county, erf, erf_final, helper, datetime.fromisoformat(erf_training_cutoff), 
                7, erf_pred_dates, final_pred_dates, len(final_pred_dates)
            )
        else:
            pred = small_NN.predict_one_county(
                county, erf, erf_final, helper, datetime.fromisoformat(erf_training_cutoff),
                7, erf_pred_dates, final_pred_dates, len(final_pred_dates)
            )
        
        return pred
        
