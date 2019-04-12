import STRING
import pandas as pd
import seaborn as sns
from statsmodels.sandbox.regression.gmm import IV2SLS
import numpy as np
import statsmodels.api as sm

sns.set()

df = pd.read_csv(STRING.final_file_hr, sep=';', encoding='latin1', date_parser=['DATE'])
df['DATE'] = pd.to_datetime(df['DATE'])
df = df[df['DATE'] > '2013-12-29']
df = df.sort_values(by='DATE', ascending=True)
df = df.set_index('DATE')
print(df.columns)

# VARIABLE Y
df['PSPAIN'] = np.log(df['PSPAIN'] + 1) - np.log(df['PNORD'] + 1)
df['QDIF'] = np.log(df['sum(TOTAL_PRODUCCION_ES)'] + 1) - np.log(df['sum(QNORD)'] + 1)

df['PSPAIN'] = (df['PSPAIN'] - df['PSPAIN'].shift(1)) - (df['PSPAIN'].shift(1) - df['PSPAIN'].shift(2))
df['QDIF'] = (df['QDIF'] - df['QDIF'].shift(1)) - (df['QDIF'].shift(1) - df['QDIF'].shift(2))

df['LQ1'] = df['QDIF'].shift(1)
df['LQ2'] = df['QDIF'].shift(2)

# STATIONARITY
for cols in df.drop(['PSPAIN', 'TREND', 'TME_MADRID', 'TMAX_MADRID',
                'PP_MADRID', 'WORKDAY', 'SUMMER', 'WINTER', 'TME_BCN', 'TMAX_BCN', 'TMIN_BCN', 'PP_BCN', 'INDEX', 'QDIF',
                'PRCP', 'TAVG', 'TMAX', 'TMIN', 'Portugal', 'Norway', 'Denmark', 'Finland', 'Sweden', 'LQ1', 'LQ2'
                     ], axis=1).columns.values.tolist():
    df['D_' + cols] = (df[cols] - df[cols].shift(1)) - (df[cols].shift(1) - df[cols].shift(2))
    del df[cols]

# AR Component
df['ar.L1.PSPAIN'] = df['PSPAIN'].shift(1)
df['ar.L2.PSPAIN'] = df['PSPAIN'].shift(2)
df['ar.L3.PSPAIN'] = df['PSPAIN'].shift(3)
df['ar.L4.PSPAIN'] = df['PSPAIN'].shift(4)
df['ar.L5.PSPAIN'] = df['PSPAIN'].shift(5)
df['ar.L6.PSPAIN'] = df['PSPAIN'].shift(6)
df['ar.L7.PSPAIN'] = df['PSPAIN'].shift(7)

# MA component
width = 1
lag = df['PSPAIN'].shift(width)
window = lag.rolling(window=width + 2)
means = pd.DataFrame(window.mean().values, columns=['ma.L1.PSPAIN'], index=df.index)
df = pd.concat([df, means], axis=1)


# X, Y
df['WORKDAY'] = (df['WORKDAY'] + 1)
df['Portugal'] = (df['Portugal'] + 1)
df['Norway'] = (df['Norway'] + 1)
df['Sweden'] = (df['Sweden'] + 1)
df['Denmark'] = (df['Denmark'] + 1)
df['Finland'] = (df['Finland'] + 1)
df['WINTER'] = (df['WINTER'] + 1)
df['SUMMER'] = (df['SUMMER'] + 1)
df['INDEX'] = df['INDEX'] + 5
df['TME'] = (df['TAVG']+0.1)**2
df['TMIN'] = (df['TMIN']+0.1)
df['TMAX'] = (df['TMAX']+0.1)
df['PRCP'] = (df['PRCP']) + 1
df['TME_MADRID'] = df['TME_MADRID']**2
df['PP*TMIN'] = df['TMAX'] * df['INDEX']


# INSTRUMENTS
instruments = ['WORKDAY', 'INDEX', 'Portugal', 'TME', 'WINTER', 'PRCP', 'TMIN', 'SUMMER', 'Norway', 'Denmark',
               'Finland', 'Sweden']
inst_square = []
for inst in instruments:
    df[inst] = np.log(df[inst])
    df[inst] = (df[inst] - df[inst].shift(1)) - (df[inst].shift(1) - df[inst].shift(2))

df = df.dropna(axis=0)

variable_used = ['D_sum(TOTAL_IMPORTACION_ES)',
                'D_NULL_PRICE', 'LQ1', 'D_sum(TOTAL_PRODUCCION_POR)', 'D_DUMMY_BACK_50_DAY', 'ar.L1.PSPAIN',
                 'ar.L2.PSPAIN', 'ar.L3.PSPAIN',
                 'ar.L4.PSPAIN', 'ar.L5.PSPAIN', 'ar.L6.PSPAIN', 'ar.L7.PSPAIN']

# QSPAIN predict
predict_qesp = sm.OLS(endog=df['D_sum(TOTAL_PRODUCCION_ES)'], exog=df[['D_sum(TOTAL_IMPORTACION_ES)',
            'D_NULL_PRICE', 'D_DUMMY_BACK_50_DAY', 'WORKDAY']]).fit()

predict_qesp = predict_qesp.predict(df[['D_sum(TOTAL_IMPORTACION_ES)',
            'D_NULL_PRICE', 'D_DUMMY_BACK_50_DAY', 'WORKDAY']])

predict_qesp = pd.DataFrame(predict_qesp, columns=['QSPAIN'])
predict_qesp['LQ1'] = predict_qesp['QSPAIN'].shift(1)

# QNORD Predict
predict_nor = sm.OLS(endog=df['D_sum(QNORD)'], exog=df[['TMIN', 'INDEX', 'Denmark']]).fit()

predict_nor = predict_nor.predict(df[['TMIN', 'INDEX', 'Denmark']])

predict_nor = pd.DataFrame(predict_nor, columns=['QNORD'])
df = pd.concat([df.drop('LQ1', axis=1), predict_nor, predict_qesp], axis=1)

mod = sm.tsa.ARMA(endog=df['PSPAIN'], exog=df[['D_sum(TOTAL_IMPORTACION_ES)',
            'LQ1', 'D_DUMMY_BACK_50_DAY', 'QSPAIN', 'D_sum(QNORD)', 'D_sum(TOTAL_PRODUCCION_POR)']],
                  order=(7, 0), missing='drop')
results = mod.fit(trend='nc')
print(results.summary())