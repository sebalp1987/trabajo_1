import STRING
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

sns.set()

df = pd.read_csv(STRING.final_file_hr, sep=';', encoding='latin1', date_parser=['DATE'])
df['DATE'] = pd.to_datetime(df['DATE'])
# df = df[df['DATE'] < '2014-02-01']
df.loc[df['DATE']>'2014-01-01', 'DUMMY_BACK_70_DAY'] = 0
df['TARGET'] = pd.Series(0, index=df.index)
df.loc[df['DATE'] <= '2013-12-31', 'TARGET'] = 1
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
'''
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
'''

# X, Y
df['WORKDAY'] = df['WORKDAY'] + 1
df['WORKDAY'] = np.log(df['WORKDAY'])-np.log((df['INDEX'] + 5))
df['TME'] = df['TME_MADRID']**2 / (df['TAVG']**2 + 0.1)


# INSTRUMENTS
instruments = ['WORKDAY', 'TME', 'WINTER']
inst_square = []
for inst in instruments:
    df[inst] = (df[inst] - df[inst].shift(1)) - (df[inst].shift(1) - df[inst].shift(2))

df = df.dropna(axis=0)

df = df[['PSPAIN',  'QDIF',
         'sum(TOTAL_IMPORTACION_ES)',
         'sum(TOTAL_DEMANDA_NAC_ES)', 'sum(TOTAL_EXPORTACIONES_ES)',
         'sum(TOTAL_DDA_ES)', 'sum(TOTAL_POT_IND_ES)',
         'sum(TOTAL_PRODUCCION_POR)', 'sum(TOTAL_DEMANDA_POR)',
         'sum(HIDRAULICA_CONVENC)', 'sum(HIDRAULICA_BOMBEO)', 'sum(NUCLEAR)',
         'sum(CARBON NACIONAL)', 'sum(CARBON_IMPO)', 'sum(CICLO_COMBINADO)',
         'sum(FUEL_SIN_PRIMA)', 'sum(FUEL_PRIMA)', 'sum(REG_ESPECIAL)', 'TREND',
         'PRICE_OIL', 'PRICE_GAS', 'RISK_PREMIUM',
         'GDP', '%EOLICA', 'PRCP', 'TMAX', 'TMIN', 'TAVG', 'WORKDAY', 'NULL_PRICE', 'SUMMER',
         'WINTER', 'DUMMY_BACK_70_DAY', 'TARGET']]

# Normal - Anormal: 5-30 days
train = df[df['TARGET'] == 0].drop(['DUMMY_BACK_70_DAY'], axis=1)
df = df[df['TARGET'] == 1]


abnormal = df[(df['DUMMY_BACK_70_DAY'] == 1)] # 5 to 30 days
normal = df[-df.isin(abnormal)].dropna()
print(normal.shape)
abnormal = abnormal.drop(['DUMMY_BACK_70_DAY'], axis=1)
normal = normal.drop(['DUMMY_BACK_70_DAY'], axis=1)
abnormal['TARGET'] = pd.Series(1, index=abnormal.index)
normal['TARGET'] = pd.Series(0, index=normal.index)
valid_normal, valid_mixed = train_test_split(normal, shuffle=None, train_size=0.5)
train.to_csv(STRING.train, sep=';', index=True)
valid_mixed.to_csv(STRING.valid_mixed, sep=';', index=True)
valid_normal.to_csv(STRING.valid_normal, sep=';', index=True)
abnormal.to_csv(STRING.test, sep=';', index=True)


