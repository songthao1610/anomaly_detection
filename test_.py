import pytest

from src.app_complete.models.Univariate import *
@pytest.mark.parametrize('df_input',[
  [1,2,3,4,5,6,7,8,9,10,90],
  [1,1,1,2,2,3,3,4,4,5,5,5,-13],
  [0.01, 0.03, 0.05,0.02,0.01,0.03,0.4],
  [1000,1500,1230,13,1780,1629,1450,1106],
  [1,2,3,4,5,6,7,8,9,10,19.4]
])

def test_detect_univariate_statistical_returns_single_anomaly(df_input):
  df = pd.DataFrame(df_input, columns = ["value"])
  sensitivity_score = 50 
  max_fraction_anomalies = 0.5
  (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
  num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
  assert(num_anomalies == 1)