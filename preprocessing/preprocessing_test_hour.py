import STRING
import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import normaltest, shapiro
import numpy as np

from resource import temporal_statistics as sts

sns.set()

df = pd.read_csv(STRING.final_file_hr, sep=';', encoding='latin1')
df['DATE_HOUR'] = pd.to_datetime(df['DATE_HOUR'], format='%d-%m-%Y %H:%M:%S')
df = df.sort_values(by='DATE_HOUR', ascending=True)
df = df.set_index('DATE_HOUR')
print(df.columns)

# REMOVE BEFORE
'''
df = df[df['DUMMY_2010_REGIMEN'] == 1]
del df['DUMMY_2010_REGIMEN']
'''

# VARIABLE Y
df['PSPAIN'] = np.log(df['PESPANIA'] + 1) - np.log(df['PNORD'] + 1)

# ESTACIONARIEDAD Y
t_value, critical_value = sts.test_stationarity(df['PSPAIN'], plot_show=True)

# No parece estacionaria, usamos la primera diferencia
df['PSPAIN'] = df['PSPAIN'] * (
            1 - df['PSPAIN'].shift(1) - df['PSPAIN'].shift(2) - df['PSPAIN'].shift(3) - df['PSPAIN'].shift(4) - df[
        'PSPAIN'].shift(5)) * (1 - df['PSPAIN'].shift(25) - df['PSPAIN'].shift(26) - df['PSPAIN'].shift(27) - df[
    'PSPAIN'].shift(28)) * (1 - df['PSPAIN'].shift(49) - df['PSPAIN'].shift(50) - df['PSPAIN'].shift(51) - df[
    'PSPAIN'].shift(52) * (1 - df['PSPAIN'].shift(168) - df['PSPAIN'].shift(169) - df['PSPAIN'].shift(170)))
df = df.dropna(subset=['PSPAIN'], axis=0)
print(df)

t_value, critical_value = sts.test_stationarity(df['PSPAIN'], plot_show=True)

# DECOMPOSE SERIE ADDITIVE (DETREND / DESEASONALIZE)
print(df['PSPAIN'])
print(df[['PSPAIN']])
result_add = seasonal_decompose(df['PSPAIN'], model='additive', extrapolate_trend='freq', freq=365)
result_add.plot().suptitle('PSPAIN', fontsize=22)
plot.show()
plot.close()

# ESTACIONARIEDAD DEMAS VARIABLES
for col in df.columns.values.tolist():
    df[col] = df[col].map(float)
bool_cols = [col for col in df
             if df[[col]].dropna().isin([0, 1]).all().values]
for cols in df.drop(['PSPAIN', 'TREND'
                     ] + bool_cols, axis=1).columns.values.tolist():
    print(cols)
    # df[cols] = np.log(df[cols]) log-linear
    df[cols] = df[cols].map(float)
    dif = 0

    try:
        t_value, critical_value = sts.test_stationarity(df[cols], plot_show=False)

        while t_value > critical_value:
            dif += 1
            t_value, critical_value = sts.test_stationarity((df[cols] - df[cols].shift(dif)).dropna(axis=0),
                                                            plot_show=False)
    except:
        pass
    if dif > 0:
        df['D' + str(dif) + '_' + cols] = df[cols] - df[cols].shift(dif)
        del df[cols]

df = df.dropna(axis=0)
df.to_csv(STRING.file_detrend_hs, index=True, sep=';')