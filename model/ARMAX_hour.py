import pandas as pd
import STRING
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plot
from scipy.stats import normaltest, shapiro
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import linear_model
import resource.temporal_statistics as sts
from config import config
sns.set()

df = pd.read_csv(STRING.file_detrend_hs, sep=';')

# Variables Used
variable_used = config.params.get('linear_var')
variables = []
for col in variable_used:
    for col_d in df.columns.values.tolist():
        if col_d.endswith(col):
            variables.append(col_d)

variable_used = variables

df['DATE_HOUR'] = pd.to_datetime(df['DATE_HOUR'], format='%Y-%m-%d %H:%M:%S')
df = df.set_index('DATE_HOUR')
df['QDIF'] = df['TOTAL_PRODUCCION_ES'] - df['QNORD']
df = df.dropna(axis=0)

variable_used += ['QDIF']
# VIF
print(variable_used)
vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(df[variable_used].values, i) for i in
              range(df[variable_used].shape[1])]
vif['features'] = df[variable_used].columns
print(vif)

y = df[['PSPAIN']]

# OLS
x_ols = df[variable_used].drop('PSPAIN', axis=1)
reg1 = sm.OLS(endog=y, exog=x_ols, missing='none')
results = reg1.fit()
print(results.summary())


# RESIDUALS
prediction_ols = results.predict(x_ols)
res = pd.DataFrame(prediction_ols, columns=['predict'])
res = pd.concat([res, y], axis=1)
res['error'] = res['PSPAIN'] - res['predict']

# RESIDUAL ESTATIONARITY
sts.test_stationarity(res['error'], plot_show=False)

# RESIDUAL SERIAL CORRELATION
sts.serial_correlation(res['error'], plot_show=False)

# We check PACF y ACF para MA y AR parameters.
fig, ax = plot.subplots(2, 1, figsize=(15, 8))
fig = sm.graphics.tsa.plot_acf(res['error'], lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res['error'], lags=50, ax=ax[1])
plot.show()
plot.close()
del res

# SARIMAX MODEL
mod = sm.tsa.ARMA(endog=y, exog=x_ols, order=(2, 2))
results = mod.fit()
print(results.summary())

# RESIDUALS
prediction = results.predict(start=9, end=len(y)-1, dynamic=True)
res = pd.DataFrame(prediction, columns=['predict'])
res = pd.concat([res, y], axis=1)
res['error'] = res['PSPAIN'] - res['predict']

# RESIDUAL ESTATIONARITY
sts.test_stationarity(res['error'], plot_show=False)

# RESIDUAL SERIAL CORRELATION
sts.serial_correlation(res['error'], plot_show=False)
