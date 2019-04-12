import pandas as pd
import STRING
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plot
from scipy.stats import normaltest, shapiro
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import linear_model
import resource.temporal_statistics as sts
from preprocessing import stepwise_reg
from config import config
sns.set()

df = pd.read_csv(STRING.file_detrend, sep=';')
print(df.columns)
# Variables Used
variable_used = config.params.get('linear_var')
variables = []
for col in variable_used:
    for col_d in df.columns.values.tolist():
        if col_d.endswith(col):
            variables.append(col_d)

variable_used = variables


df = df.set_index('DATE')

df['LQ1'] = df['QDIF'].shift(1)
df['LQ2'] = df['QDIF'].shift(2)
df['LQ3'] = df['QDIF'].shift(3)
df['C'] = pd.Series(1, index=df.index)
df = df.dropna(axis=0)

variable_used += ['LQ1', 'LQ2', 'LQ3']



# X, Y
y = df[['PSPAIN']]
x_ols = df[variable_used].drop(['PSPAIN'], axis=1)
vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(x_ols.values, i) for i in
              range(x_ols.shape[1])]
vif['features'] = x_ols.columns
print(vif)
#x_ols = x_ols.drop(['TREND', 'sum(REG_ESPECIAL)', 'sum(TOTAL_IMPORTACION_ES)', 'sum(HIDRAULICA_BOMBEO)'], axis=1)

vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(x_ols.values, i) for i in
              range(x_ols.shape[1])]
vif['features'] = x_ols.columns
print(vif)



# 4) STEPWISE REGRESION
names = x_ols.columns
variables = stepwise_reg.stepwise_regression.setpwise_reg(x_ols.values.tolist(), y.values.tolist(), names)
var_dummy = [var for var in x_ols.columns.values.tolist() if var.startswith('D_DUMMY')]
print('ACA', var_dummy[0])
if var_dummy[0] not in variables:
    variables.append(var_dummy[0])
print(variables)
x_ols = x_ols[variables]
reg1 = sm.OLS(endog=y, exog=x_ols, missing='none')
results = reg1.fit()
print(results.summary())


# RESIDUALS
prediction_ols = results.predict(x_ols)
res = pd.DataFrame(prediction_ols, columns=['predict'])
res = pd.concat([res, y], axis=1)
res['error'] = res['PSPAIN'] - res['predict']


#  5) RESIDUAL ESTATIONARITY
sts.test_stationarity(res['error'], plot_show=False)

# 6) RESIDUAL SERIAL CORRELATION
sts.serial_correlation(res['error'], plot_show=False)

# 7) NORMALIDAD RESIDUOS
plot.hist(res['error'])
# plot.show()
plot.close()

alpha = 0.05
stat, p = shapiro(res['error'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
stat, p = normaltest(res['error'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

# 8) We check PACF y ACF para MA y AR parameters.
fig, ax = plot.subplots(2, 1, figsize=(15, 8))
fig = sm.graphics.tsa.plot_acf(res['error'], lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res['error'], lags=50, ax=ax[1])
# plot.show()
plot.close()
del res

# SARIMAX MODEL: Ir porbando de a uno. 9) Remover no significativas

print(x_ols.columns)
'''
x_ols = x_ols.drop(['D_NULL_PRICE', 'D_sum(CARBON NACIONAL)', 'LQ2'
                    ], axis=1)

'''
# BEST MODEL

x_ols = df[['D_sum(QNORD)', 'D_sum(TOTAL_PRODUCCION_ES)', 'D_sum(TOTAL_IMPORTACION_ES)',
            'D_NULL_PRICE', 'LQ1', 'D_sum(TOTAL_PRODUCCION_POR)'
            ] + [var_dummy[0]]]
'''
best_aic = []
ar_ma = []
for ar in range(1, 10, 1):
    for ma in range(0, 10, 1):
        # VAR model to evaluate AIC best lags
        mod = sm.tsa.ARMA(endog=y, exog=x_ols, order=(ar, ma),
                          missing='none')
        try:
            results = mod.fit(disp=False)
            std_error = results.bse
            results = results.aic
            for se in range(0, len(std_error)-1, 1):
                if np.isnan(std_error[se]):
                    results = 999999
        except:
            results = 999999
        ar_ma.append([ar, ma])
        best_aic.append(results)

print(best_aic)
print(ar_ma)
ar_ma_coef = ar_ma[best_aic.index(min(best_aic))]
print('ARMA COEFFICIENTS', ar_ma_coef)
'''
ar_ma_coef = [7, 0]
'''
x_ols['ar.L2.PSPAIN'] = y['PSPAIN'].shift(2)
x_ols['ar.L4.PSPAIN'] = y['PSPAIN'].shift(4)
x_ols['ar.L6.PSPAIN'] = y['PSPAIN'].shift(6)
'''
x_ols = x_ols.dropna(axis=0)
y = y[x_ols.index[0]::]

mod = sm.tsa.ARMA(endog=y, exog=x_ols, order=(ar_ma_coef[0], ar_ma_coef[1]), missing='drop')
results = mod.fit(trend='nc')
print(results.summary())

# RESIDUALS
y = y.reset_index(drop=True)
prediction = results.predict(start=0, end=y.index[len(y)-1])
res = pd.DataFrame(prediction, columns=['predict']).reset_index(drop=True)
print('ARMA R2', r2_score(y, res['predict'].values))
res = pd.concat([res, y], axis=1)
res['error'] = res['PSPAIN'] - res['predict']

# RESIDUAL ESTATIONARITY
print('dw test', sm.stats.stattools.durbin_watson(res['error'], axis=0))
sts.test_stationarity(res['error'], plot_show=True)

# RESIDUAL SERIAL CORRELATION
sts.serial_correlation(res['error'], plot_show=True)
fig, ax = plot.subplots(2, 1, figsize=(15, 8))
fig = sm.graphics.tsa.plot_acf(res['error'], lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res['error'], lags=50, ax=ax[1])
plot.show()
plot.close()

# NORMALIDAD RESIDUOS
sns.distplot(res['error'], hist=True, kde=True, color = 'darkblue',
             hist_kws={'edgecolor': 'black'},
             kde_kws={'linewidth': 4})
plot.show()
plot.close()

alpha = 0.05
stat, p = shapiro(res['error'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
stat, p = normaltest(res['error'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

# 10) HAY MULTICOLINEALIDAD?
# VIF: sum(REG_ESPECIAL)
vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(x_ols.values, i) for i in
              range(x_ols.shape[1])]
vif['features'] = x_ols.columns
print(vif)
