import pandas as pd 
import math
from pandas.core import base 
from statsmodels import robust
import numpy as np
from scipy.stats import shapiro, normaltest, anderson, boxcox
import scikit_posthocs as ph 
from sklearn.mixture import GaussianMixture



def check_stat(val, midpoint, distance, n):
  if (abs(val-midpoint) < (n*distance)):
    return abs(val-midpoint)/(n*distance)
  else:
    return 1.0

def check_sd(val, mean, sd, min_num_sd):
  return check_stat(val, mean, sd, min_num_sd)

def check_mad(val, median, mad, min_num_mad):
  return check_stat(val, median, mad, min_num_mad)

#iqr is the differrence between the 75th and 25th percentiles of a dataset
def check_iqr(val, median, p25, p75, iqr, min_iqr_diff):
  if (val < median):
    if (val > p25):
      return 0.0
    # if the value is between p25 and the outlier break point,
    # return a fractional score representing how distant it is
    elif (val > p25 - (min_iqr_diff * iqr)):
      return abs(p25 - val)/(min_iqr_diff * iqr)
    else:
      return 1.0
  else: 
    if (val < p75):
      return 0.0 
    elif (val < p75 + (min_iqr_diff * iqr)):
      return abs(val-p75) / (min_iqr_diff * iqr)
    else:
      return 1
    
def run_tests(df):
  mean = df['value'].mean()
  sd = df['value'].std()
  p25 = np.quantile(df['value'], 0.25)
  p75 = np.quantile(df['value'], 0.75)
  iqr = p75 - p25
  median = df['value'].median()
  mad = robust.mad(df['value'])  
  calculations = {'mean': mean, "sd":sd, "p25":p25, 
                  "median": median, "p75": p75, "iqr": iqr, "mad": mad}
  # for each test, execute and add a new score 
  df['sds'] = [check_sd(val, mean,sd, 3.0) for val in df['value']]
  df['mads'] = [check_mad(val, median, mad, 3.0) for val in df['value']]
  df['iqrs'] = [check_iqr(val, median, p25, p75, iqr, 1.5) for val in df['value']]
  return (df, calculations)

def score_results(
    df, 
    weights
):
  return df.assign(anomaly_score=(
    df['sds'] * weights['sds'] +
    df['iqrs'] * weights['iqrs'] +
    df['mads'] * weights['mads']
  ))
    

def determine_outliers(
    df, 
    sensitivity_score,
    max_fraction_anomalies
):
  sensitivity_score = (100 - sensitivity_score) / 100.0
  max_fraction_anomaly_score = np.quantile(df['anomaly_score'],
                                           1.0 - max_fraction_anomalies)
  if max_fraction_anomaly_score > sensitivity_score and max_fraction_anomalies < 1.0:
    sensitivity_score = max_fraction_anomaly_score
    return df.assign(is_anomaly = (df['anomaly_score']>sensitivity_score))
  
def detect_univariate_statistical(
    df,
    sensitivity_score,
    max_fraction_anomalies
):
  weights = {"sds": 0.25, "iqrs": 0.35, "mads": 0.45}
  if (df['value'].count() < 3):
    return (df.assign(is_anomaly = False, anomaly_score = 0.0), 
            weights, "Must have a minimum of at least three data points for anomaly detection.")
  elif (max_fraction_anomalies <= 0.0 or max_fraction_anomalies > 1.0):
    return (df.assign(is_anomaly=False, anomaly_score = 0.0), 
            weights, "Must have a valid max fraction of anomalies, 0 < x <=1.0" )
  elif (sensitivity_score <= 0 or sensitivity_score > 100):
    return (df.assign(is_anomaly=False, anomaly_score = 0.0), 
            weights, "Must have a valid sensitivity score, 0 < x <= 100")
  else:
    df_tested,calculations = run_tests(df)
    df_scored = score_results(df_tested, weights)
    df_out = determine_outliers(df_scored, sensitivity_score, max_fraction_anomalies)
    return (df_out, weights, {"message": "Ensemble of [mean +/- 3*SD, median +/- 1.5*IQR, median +/- 3*MAD].",
                              "calculations": calculations})