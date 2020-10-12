from scipy.special import erf
from scipy.optimize import curve_fit
import numpy as np
import time
import pandas as pd

################################################################
############# Util class used throughout the file ##############
################################################################

class utils:
    def get_processed_df(file_name='nyt_us_counties.csv'):
        homedir = "../"
        datadir = f"{homedir}/data/us/covid/"
        df = pd.read_csv(datadir + file_name)
        df['date_processed'] = pd.to_datetime(df['date'].values)
        # Convert YYYY-MM-DD date format into integer number of days since the first day in the data set
        df['date_processed'] = (df['date_processed'] - df['date_processed'].min()) / np.timedelta64(1, 'D')
        # Special correction for the NYT data set
        df.loc[df['county'] == 'New York City', 'fips'] = 36061.
        return df


    def process_date(date_str, df):
        return (pd.to_datetime(date_str) - pd.to_datetime(df['date'].values).min()) / np.timedelta64(1, 'D')


    def get_region_data(df, county_fips, proc_date=None, key='deaths'):
        county_fips = float(county_fips)
        d = df.loc[df['fips'] == county_fips]
        if proc_date is not None:
            vals = d.loc[d['date_processed'] == proc_date][key].values
            if len(vals) == 0:
                return 0.0
            return vals[0]
        return d


    def all_output_dates():
        ret = ['2020-04-%02d' % x for x in range(1, 31)]
        ret += ['2020-05-%02d' % x for x in range(1, 32)]
        ret += ['2020-06-%02d' % x for x in range(1, 31)]
        return ret


    def all_fips_in_df(df):
        # Get a sorted list of all FIPS string codes in a dataframe
        return sorted(list(set(['%d' % x for x in df['fips'].values if not np.isnan(x)])))


    def all_output_fips(sample_out_file):
        # Get a sorted list of all FIPS codes in the sample output file
        homedir = "../"
        datafile = f"{homedir}/" + sample_out_file
        all_data = np.genfromtxt(datafile, delimiter=',', dtype='str')
        all_fips = set([x.split('-')[-1] for x in all_data[1:, 0]])
        return sorted(list(all_fips)), all_data[1:, 0]


    def fill_missing_dates(t, y):
        # If a time series is missing days, fill those missing days with a copy of the most recent value
        ret_t = np.arange(np.min(t), np.max(t) + 1)
        ret_y = np.zeros(len(ret_t))
        dat_ind = 0
        for ret_ind in range(len(ret_t)):
            if ret_t[ret_ind] in t:
                ret_y[ret_ind] = y[dat_ind]
                dat_ind += 1
            else:
                ret_y[ret_ind] = ret_y[ret_ind - 1]
        return ret_t, ret_y



################################################################
############### General Curve Fitting Functions ################
################################################################

def erf_curve(t, log_max, slope, center):
    '''
    t: array of time values to input to the erf function
    log_max, slope, center: parameters of the erf curve
    '''
    # Using log(max) as the input rather than just max makes it easier for a curve fitter to match exponential data
    max_val = 10 ** log_max
    deaths = max_val * (1 + erf(slope * (t - center))) / 2
    return deaths


def lin_curve(t, slope, intercept):
    '''
    t: array of time values to input to the linear function
    slope, intercept: parameters of the line
    '''
    ret = t * slope + intercept
    return ret

def get_time_list(data, future=0):
    '''
    data: general dataframe, used to find the first date in history
    future: number of days to extend the time values past present day
    '''
    t = data['date_processed'].values
    t = np.arange(np.min(t), np.max(t) + 1)  # Fill in any potential missing days
    if future > 0:  # Add on days in the future
        extrapolation = np.arange(future)
        t = np.concatenate((t, extrapolation + t[-1] + 1))
    return t





################################################################
####################### Model Functions #######################
################################################################

def run_model(func, params, t):
    '''
    func: method handle being run
    params: parameters to feed to the model
    t: input time values to the model
    '''
    preds = func(t, *params)
    preds[preds < 0] = 0  # Remove spurious negative death predictions

    return preds

def make_erf_point_predictions(df, county_fips, key='deaths', last_date_pred='2020-06-30', start_date='2020-03-31',
                               boundary_date=None):
    '''
    df: main nyt data frame
    county_fips: fips code of the county to be fit
    key: 'deaths' for COVID-19 deaths, 'cases' for COVID-19 confirmed cases
    last_date_pred: last day to make predictions for. If 'None', stop at current day
    start_date: first date to list fitted values for. If 'None', start at beginning of dataframe. If do_diff is True,
        this should be one day before the first day you want difference values for
    boundary_date: date at which to cut off data used for fitting
    do_diff: if true, report the daily increase in cases/deaths rather than cumulative values
    '''
    num_days = int(utils.process_date(last_date_pred, df) - utils.process_date(start_date, df))
    data = utils.get_region_data(df, county_fips)
    if len(data) == 0:  # If there's no data for this FIPS, just return zeroes
        return np.zeros(num_days)
    first_date_obv_proc = np.min(data['date_processed'].values)
    boundary = None if boundary_date is None else int(utils.process_date(boundary_date, df) - first_date_obv_proc + 1)

    x = data['date_processed'].values[:boundary]
    if len(x) == 0:  # If there's no data for this FIPS, just return zeroes
        return np.zeros(num_days)
    if start_date is None:
        start_date_proc = first_date_obv_proc
    else:
        start_date_proc = utils.process_date(start_date, df)
    last_date_obv_proc = np.max(x)
    if last_date_pred is None:
        last_date_pred_proc = last_date_obv_proc
    else:
        last_date_pred_proc = utils.process_date(last_date_pred, df)

    y = data[key].values[:boundary]
    if np.max(y) == 0:  # If all data we have for this FIPS is zeroes, just return zeroes
        return np.zeros(num_days)
    thresh_y = y[y >= 10]  # Isolate all days with at least 10 cases/deaths
    # If we have fewer than 5 days with substantial numbers of cases/deaths there isn't enough information to do an
    # erf fit, so just do a simple linear fit instead
    do_lin_model = len(thresh_y) < 5
    if do_lin_model:
        fit_func = lin_curve
        # Perform a linear fit on the latest 5 days of data
        fit_x, fit_y = x[-5:], y[-5:]
        # Pad with zeroes if we have fewer than 5 days of data
        if len(fit_x) < 5:
            fit_x = np.concatenate((np.zeros(5 - len(fit_x)), fit_x))
            fit_y = np.concatenate((np.zeros(5 - len(fit_y)), fit_y))
        fit_params0 = [0, 0]
        # The slope should be at least 0 and at most the largest 1-day increase
        # The intercept can be very low but shouldn't be above the minimum data value
        fit_bounds = [[0, -100 * np.max(y)], [max(1, np.max(np.diff(fit_y))), np.min(y)]]
    else:
        fit_func = erf_curve
        fit_x, fit_y = x, y
        fit_params0 = [np.log10(2 * np.max(data[key])), 0.1, 30]
        # The max value should be between the current max and 100x the current max
        # The slope was given a wide range around common values
        # The infection shouldn't peak before the data started or after the end of ~July
        fit_bounds = [bnd for bnd in zip(*[[np.log10(np.max(data[key])), np.log10(100 * np.max(data[key]))],
                                           [0.001, 10],
                                           [0, 200]])]
    # Use scipy to fit either a linear or erf model to the data
    popt, pcov = curve_fit(fit_func, fit_x, fit_y,
                           p0=fit_params0, bounds=fit_bounds)
    t = np.arange(start_date_proc, last_date_pred_proc + 1)
    return np.diff(run_model(fit_func, popt, t))






################################################################
######################### Run the Model ########################
################################################################

class BenchmarkModel:
    def predict_all_counties(self, last_date_pred, boundary_date):
        df = utils.get_processed_df()
        out_fips, all_row_starts = utils.all_output_fips('sample_submission.csv')
        
        preds_map = {}
        for fi, fips in enumerate(out_fips):
            pred = make_erf_point_predictions(df, fips, last_date_pred=last_date_pred, boundary_date=boundary_date,
                                               start_date=boundary_date, key='deaths')
            
            preds_map[fips] = pred
            
        return preds_map