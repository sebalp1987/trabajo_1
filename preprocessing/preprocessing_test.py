import STRING
import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

import numpy as np

from resource import temporal_statistics as sts

sns.set()

df = pd.read_csv(STRING.final_file_hr, sep=';', encoding='latin1', date_parser=['DATE'])
df['DATE'] = pd.to_datetime(df['DATE'])
df = df[df['DATE'] < '2014-02-01']
df = df.sort_values(by='DATE', ascending=True)
df = df.set_index('DATE')
print(df.columns)

# VARIABLE Y
df['PSPAIN'] = np.log(df['PSPAIN'] + 1) - np.log(df['PNORD'] + 1)
df['QDIF'] = np.log(df['sum(TOTAL_PRODUCCION_ES)'] + 1) - np.log(df['sum(QNORD)'] + 1)

# 1) ESTACIONARIEDAD Y: TEST DF Var Y
t_value, critical_value = sts.test_stationarity(df['PSPAIN'], plot_show=True)

# DECOMPOSE SERIE ADDITIVE (DETREND / DESEASONALIZE)
result_add = seasonal_decompose(df['PSPAIN'], model='additive', extrapolate_trend='freq')
result_add.plot().suptitle('PSPAIN', fontsize=22)
plot.show()
plot.close()

# No parece estacionaria, usamos la primera diferencia, parece haber un efecto seasonal fuerte 7
df['PSPAIN'] = (df['PSPAIN'] - df['PSPAIN'].shift(1)) - (df['PSPAIN'].shift(1) - df['PSPAIN'].shift(2)) # Now is almost stationart
df['QDIF'] = (df['QDIF'] - df['QDIF'].shift(1)) - (df['QDIF'].shift(1) - df['QDIF'].shift(2)) # Now is almost stationart

df = df.dropna(subset=['PSPAIN'], axis=0)

t_value, critical_value = sts.test_stationarity(df['PSPAIN'], plot_show=True)


# 2) ESTACIONARIEDAD DEMAS VARIABLES + 3) GRANGER CAUSALITY
for col in df.columns.values.tolist():
    df[col] = df[col].map(float)
bool_cols = [col for col in df
             if df[[col]].dropna().isin([0, 1]).all().values]
for cols in df.drop(['PSPAIN', 'TREND', 'TME_MADRID', 'TMAX_MADRID',
                'PP_MADRID', 'WORKDAY', 'SUMMER', 'WINTER', 'TME_BCN', 'TMAX_BCN', 'TMIN_BCN', 'PP_BCN', 'INDEX', 'QDIF',
                'PRCP', 'TAVG', 'TMAX', 'TMIN', 'Portugal', 'Norway', 'Denmark', 'Finland', 'Sweden'
                     ], axis=1).columns.values.tolist():

    # df[cols] = np.log(df[cols]) log-linear
    df[cols] = df[cols].map(float)
    dif = 0
    df[cols] = np.log(df[cols] + 1)
    # DFA
    try:
        t_value, critical_value = sts.test_stationarity(df[cols], plot_show=False)

        while t_value > critical_value:
            dif += 1
            t_value, critical_value = sts.test_stationarity((df[cols] - df[cols].shift(dif)).dropna(axis=0),
                                                            plot_show=False)
    except:
        pass

    # GRENGER
    '''
    m = 1 # maximum order of integration for the group of time-series (from the differentation)
    lag = 0
    best_aic = []
    endog = df[cols]
    for lag in range(1, 10, 1):
        # VAR model to evaluate AIC best lags
        exog = pd.DataFrame(index=df.index)
        exog['AR_' + str(lag)] = df[cols].shift(lag)
        exog = exog.dropna(axis=0)
        endog = endog[1::]
        regr = OLS(endog=endog, exog=add_constant(exog)).fit()
        best_aic.append([regr.aic])

    max_aic_lag = best_aic.index(max(best_aic)) + 1
    sts.granger_causality(df, cols, dependant_variable='PSPAIN', maxlag=max_aic_lag + m)
    '''
    df['D_' + cols] = (df[cols] - df[cols].shift(1)) - (df[cols].shift(1) - df[cols].shift(2))
    del df[cols]


df = df.dropna(axis=0)
df.to_csv(STRING.file_detrend, index=True, sep=';')