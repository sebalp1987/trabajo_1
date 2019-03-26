import STRING
import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import normaltest, shapiro
import numpy as np

from resource import temporal_statistics as sts

sns.set()

df = pd.read_csv(STRING.final_file, sep=';', encoding='latin1', date_parser=['DATE'])
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values(by='DATE', ascending=True)
df = df.set_index('DATE')
print(df.columns)

# ESTACIONARIEDAD Y
t_value, critical_value = sts.test_stationarity(df['PSPAIN'], plot_show=True)

# No parece estacionaria, usamos la primera diferencia
df['PSPAIN'] = df['PSPAIN'] - df['PSPAIN'].shift(1)
df = df.dropna(subset=['PSPAIN'], axis=0)

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
df.to_csv(STRING.file_detrend, index=True, sep=';')