import pandas as pd 
import math
from pandas.core import base 
from statsmodels import robust
import numpy as np
from scipy.stats import shapiro, normaltest, anderson, boxcox
import scikit_posthocs as ph 
from sklearn.mixture import GaussianMixture



def check_stat(val, mid_point, distance, n):
  if abs(val-mid_point)< distance*n:
    return abs(val-mid_point)/(distance*n)
  else:
    return 1
  
def check_sd(val, mean, sd, min_num_sd):
  return check_stat(val, mean, sd, min_num_sd)

def check_mad(val, median, mad, min_num_mad):
  return check_stat(val, median, mad, min_num_mad)


def check_iqr(val, median, p25, p75, iqr, min_iqr_diff):
  if val < median:
    if val > p25:
      return 0
    elif val > (p25 - iqr * min_iqr_diff):
      return abs(val - p25)/(iqr * min_iqr_diff)
    else:
      return 1 
  else:
    if val < p75:
      return 0 
    elif val < (p75 + iqr * min_iqr_diff):
      return abs(val - p75)/(iqr * min_iqr_diff)
    else:
      return 1
    


def run_tests(df):
  base_calculations = perform_statistical_calculations(df['value'])
  diagnostics = { "Base calculations": base_calculations }
  (use_fitted_results, fitted_data, normalization_diagnostics) = perform_normalization(base_calculations,df)
  b = base_calculations
  df['sds'] = [check_sd(val, b['mean'], b['sd'],3) for val in df['value']]
  df['mad'] = [check_mad(val, b['median'],b['mad'],3) for val in df['value']]
  df['iqr'] = [check_iqr(val,b['median'],b['p25'],b['p75'],b['iqr'],1.5) for val in df['value']]
  tests_run = {
    'sds' : 1,
    'mads': 1,
    'iqrs': 1,
    'grubbs':0,
    'gesd':0,
  }
  # Start off with values of -1, if we run a test, we'll populate it with a valid value
  df['grubbs'] = -1 
  df['gesd'] = -1 
  # Grubbs, GESD all require that the input data be normally distributed
  # grubs requires at least 7 observations 
  # gesd requires at least 15 observations
   

  if (use_fitted_results):
    df['fitted_value'] = fitted_data
    col = df['fitted_value']
    c = perform_statistical_calculations(col)
    diagnostics['Fitted calculations'] =c 
    
    if (b['len']>= 7):
      df['grubbs'] = check_grubbs(col)
      tests_run['grubbs'] = 1
    else:
      diagnostics['Grubb Test'] = f'Did not run Grubbs test because we need at least 7 observations but only {b["len"]}'
    
    if (b['len'] >= 15):
      max_num_outliers = math.floor(b['len']/3)
      df['gesd'] = check_gesd(col, max_num_outliers)
      tests_run['gesd'] = 1
  else:
    diagnostics['Extended tests'] = "Did not run extended tests because the dataset \
      was not normal and could not be normalized"
  return (df, tests_run, diagnostics)

def score_results(df, tests_run, weights):
  tested_weights = {w: weights.get(w,0) * tests_run.get(w,0) for w in set(weights).union(tests_run)}
  max_weight = sum([tested_weights[w] for w in tested_weights])
  return df.assign(anomaly_score=(
    df['sds'] * weights['sds'] + 
    df['iqrs'] * weights['iqrs'] + 
    df['mads'] * weights['mads'] +
    df['grubbs'] * tested_weights['grubbs'] +
    df['gesd'] * tested_weights['gesd']
  )/(max_weight * 0.95))
    

def determine_outliers(
    df, 
    sensitivity_score,
    max_fraction_anomalies
):
  sensitivity_score = (100 - sensitivity_score) / 100.0
  max_fraction_anomaly_score = np.quantile(df['anomaly_score'], 1.0 - max_fraction_anomalies)
  if max_fraction_anomaly_score > sensitivity_score and max_fraction_anomalies < 1.0:
    sensitivity_score = max_fraction_anomaly_score
    return df.assign(is_anomaly=(df['anomaly_score']>sensitivity_score))
  else:
    return df.assign(is_anomaly=(df['anomaly_score']>sensitivity_score))

  
def detect_univariate_statistical(
    df,
    sensitivity_score,
    max_fraction_anomalies
):
  weights = {"sds": 0.25, "iqrs": 0.35, "mads": 0.45}
  if (df['value'].count() < 3):
    return (df.assign(is_anomaly=False,anomaly_score=0.0), 
            weights, "Must have a minimum of at least three data points for anomaly detection.")
  elif (max_fraction_anomalies <= 0.0 or max_fraction_anomalies > 1.0):
    return (df.assign(is_anomaly=False,anomaly_score=0.0), 
            weights, "Must have a valid max fraction of anomalies, 0 < x <=1.0" )
  elif (sensitivity_score <= 0 or sensitivity_score > 100):
    return (df.assign(is_anomaly=False,anomaly_score=0.0), 
            weights, "Must have a valid sensitivity score, 0 < x <= 100")
  else:
    df_tested,calculations = run_tests(df)
    df_scored = score_results(df_tested, weights)
    df_out = determine_outliers(df_scored, sensitivity_score, max_fraction_anomalies)
    return (df_out, weights, {"message": "Ensemble of [mean +/- 3*SD, median +/- 1.5*IQR, median +/- 3*MAD].",
                              "calculations": calculations})
  
def check_shapiro(col, alpha = 0.05):
  return check_basic_normal_test(col, alpha, ' Shaprio-Wilk test', shapiro)
def check_dagostino(col, alpha = 0.05):
  return check_basic_normal_test(col, alpha, "D'Agostino's K^2 test", normaltest)
  
def check_basic_normal_test(col, alpha, name, f):
  stat, p = f(col)
  return ((p > alpha),(f"{name} test, W={stat}, p= {p}, alpha = {alpha}."))

def check_anderson(col):
  anderson_normal = True
  return_str = 'Anderson-Darling test'

  result = anderson(col)
  return_str = return_str + f'Result Statistic: {result.statistic}'
  for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv: 
      return_str = return_str + f"Significance level {sl}: Critical Value = {cv}, looks normally distributed"
    else:
      anderson_normal = False
      return_str = return_str + f'Significance level {sl}: Critical Value {cv}, does not look normally distributed'
  return (anderson_normal, return_str)

def is_normally_distributed(col):
  alpha = 0.05
  if col.shape[0] < 5000:
    (shapiro_normal, shapiro_exp) = check_shapiro(col, alpha)
  else:
    shapiro_normal = True 
    shapiro_exp = f'Shapiro-Wilk test did not run because n >= 5k, n = {col.shape[0]}'
  
  if col.shape[0] >= 8:
    (dagostino_normal, dagostino_exp) = check_dagostino(col, alpha)
  else:
    dagostino_normal = True 
    dagostino_exp = f"D'Agostino's Test did not run because n < 8, n= {col.shape[0]}"
  
  (anderson_normal, anderson_exp) = check_anderson(col)

  diagnostics = {"Shapiro-Wilk": shapiro_exp, "D'Agostino":dagostino_exp,
                 "Anderson-Darling": anderson_exp}
  return (shapiro_normal and dagostino_normal and anderson_normal, diagnostics)

def normalize(col):
  l = col.shape[0]
  col_sort = sorted(col)
  col80 = col_sort[math.floor(.1 * l) + 1: math.floor(.9 * l)]
  temp_data, fitted_lambda = boxcox(col80)
  fitted_data = boxcox(col, fitted_lambda)
  return (fitted_data, fitted_lambda)

# function to perform normalization on a dataset
def perform_normalization(base_calculations, df):
  use_fitted_results = False
  fitted_data = None 
  (is_naturally_normal, natural_normality_checks) = is_normally_distributed(df['value'])
  diagnostics = {"Initial normality checks": natural_normality_checks}
  if is_naturally_normal:
    fitted_data = df['value']
    use_fitted_results = True
  if ((not is_naturally_normal) 
      and base_calculations['min'] < base_calculations['max']
      and base_calculations['min'] > 0
      and df['value'].shape[0] >= 8):
    (fitted_data, fitted_lambda) = normalize(df['value'])
    (is_fitted_normal, fitted_normality_checks) = is_normally_distributed(fitted_data)
    use_fitted_results = True 
    diagnostics['Fitted Lambda'] = fitted_lambda 
    diagnostics['Fitted normality checks'] = fitted_normality_checks
  else:
    has_variance = base_calculations['min'] < base_calculations['max']
    all_gt_zero = base_calculations['min'] > 0
    enough_observations = df['value'].shape[0] >= 8 
    diagnostics['Fitting status'] = f'Elided for space'
  return (use_fitted_results, fitted_data, diagnostics)
  
def check_grubbs(col):
  out = ph.outliers_grubbs(col)
  return find_differences(col, out)

def check_gesd(col, max_num_outliers):
  out = ph.outliers_gesd(col, max_num_outliers)
  return find_differences



def find_differences(col, out):
  #Convert column and output to sets to see what's missing
  scol = set(col)
  sout = set(out)
  sdiff = scol - sout

  res =[0.0 for val in col]
  for val in sdiff:
     indexes = col[col == val].index
     for i in indexes: 
       res[i] = 1.0
  return res

def perform_statistical_calculations(col):
  mean = col.mean()
  sd = col.std()
  p25 = np.quantile(col, 0.25)
  p75 = np.quantile(col, 0.75)
  iqr = p75 - p25
  median = col.median()
  mad = robust.mad(col)
  min = col.min()
  max = col.max()
  len = col.shape[0]
  return {'mean': mean, 'sd':sd, 'min':min,
          'max':max, 'p25':p25, 'median':median, 
          'p75': p75, 'iqr':iqr, 'mad':mad, 'len':len}
