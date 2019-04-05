import STRING
import pandas as pd
import seaborn as sns
from statsmodels.sandbox.regression.gmm import IV2SLS
import numpy as np


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
for cols in ['sum(CICLO_COMBINADO)', 'sum(FUEL_PRIMA)', 'sum(HIDRAULICA_CONVENC)']:
    df[cols] = np.log(df[cols] + 1)
    df['D1_' + cols] = df[cols] - df[cols].shift(1)

# AR Component
df['ar.L1.PSPAIN'] = df['PSPAIN'].shift(1)
df['ar.L2.PSPAIN'] = df['PSPAIN'].shift(2)
df['ar.L4.PSPAIN'] = df['PSPAIN'].shift(4)
df['ar.L6.PSPAIN'] = df['PSPAIN'].shift(6)

# MA component
width = 1
lag = df['PSPAIN'].shift(width)
window = lag.rolling(window=width + 2)
means = pd.DataFrame(window.mean().values, columns=['ma.L1.PSPAIN'], index=df.index)
df = pd.concat([df, means], axis=1)


# X, Y
df['WORKDAY'] = df['WORKDAY'] + 1
df['WORKDAY'] = np.log(df['WORKDAY'])-np.log((df['INDEX'] + 5))
df['TME'] = df['TME_MADRID']**2 / (df['TAVG']**2 + 0.1)

df = df[['PSPAIN', 'D1_sum(CICLO_COMBINADO)', 'D1_sum(FUEL_PRIMA)',
         'D1_sum(HIDRAULICA_CONVENC)', 'ma.L1.PSPAIN', 'ar.L1.PSPAIN', 'ar.L2.PSPAIN',
         'ar.L4.PSPAIN', 'ar.L6.PSPAIN', 'QDIF', 'WORKDAY', 'TME', 'WINTER', 'LQ1', 'LQ2'] + ['DUMMY_BACK_3_DAY',
                                                                                              'DUMMY_BACK_5_DAY',
                                                                                              'DUMMY_BACK_10_DAY',
                                                                                              'DUMMY_BACK_15_DAY',
                                                                                              'DUMMY_BACK_20_DAY',
                                                                                              'DUMMY_BACK_25_DAY',
                                                                                              'DUMMY_BACK_30_DAY',
                                                                                              'DUMMY_BACK_45_DAY',
                                                                                              'DUMMY_FORW_3_DAY',
                                                                                              'DUMMY_FORW_5_DAY',
                                                                                              'DUMMY_FORW_10_DAY',
                                                                                              'DUMMY_FORW_15_DAY',
                                                                                              'DUMMY_FORW_20_DAY',
                                                                                              'DUMMY_FORW_25_DAY',
                                                                                              'DUMMY_FORW_30_DAY',
                                                                                              'DUMMY_FORW_45_DAY',
                                                                                              'DUMMY']]

# INSTRUMENTS
instruments = ['WORKDAY', 'TME', 'WINTER']
inst_square = []
for inst in instruments:
    df[inst] = (df[inst] - df[inst].shift(1)) - (df[inst].shift(1) - df[inst].shift(2))

df = df.dropna(axis=0)
variable_used = ['D1_sum(CICLO_COMBINADO)', 'LQ1', 'LQ2',
                 'D1_sum(HIDRAULICA_CONVENC)', 'DUMMY_FORW_45_DAY', 'ma.L1.PSPAIN', 'ar.L1.PSPAIN', 'ar.L2.PSPAIN',
                 'ar.L4.PSPAIN', 'ar.L6.PSPAIN']

variable_instrumented = ['QDIF']
df[variable_used + variable_instrumented + instruments + ['PSPAIN']].to_csv('test.csv', sep=';')
y = df[['PSPAIN']]
reg = IV2SLS(endog=y, exog=df[variable_used + variable_instrumented], instrument=df[variable_used + instruments])
results = reg.fit()
print(results.summary())