import pandas as pd
import STRING
from linearmodels import RandomEffects
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from datetime import timedelta

back = 'BACK_'
dummy = 75

df = pd.read_csv(STRING.file_panel_data, sep=';')
df['DATE2'] = df['DATE']
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')
df['WEEKDAYS'] = df['DATE'].dt.dayofweek
df['MONTHS'] = df['DATE'].dt.month

dummy_day = pd.get_dummies(df['WEEKDAYS'], prefix='D_DAY', drop_first=True)
dummy_month = pd.get_dummies(df['MONTHS'], prefix='D_MONTH', drop_first=True)
df = pd.concat([df, dummy_day, dummy_month], axis=1)

df = df.set_index(['MARKET', 'YEAR', 'DATE_DDD'])
df['T'] = df['FE_SPAIN']
df['S'] = df['DUMMY_' + back + str(dummy) + '_DAY']
df['G'] = df['TIMEEF_2013']

df['S*G'] = df['S'] * df['G']
df['T*G'] = df['T'] * df['G']
df['T*S'] = df['T'] * df['S']

df['T*S*G'] = df['T'] * df['S'] * df['G']

exog_vars = ['Q', 'TME', 'PP', 'PRICE_OIL', 'PRICE_GAS', 'AR1', 'AR2', 'AR3',
             'WINTER', 'NULL_PRICE', 'LITINIT', 'T', 'S', 'G', 'S*G', 'T*G',
             'T*S', 'T*S*G', 'WORKDAY', 'DATE2', 'TREND'] + [x for x in df.columns if x.startswith('D_DAY')]
df['TME'] = df['TME'] + 39
for col in exog_vars:
    if col not in ['WORKDAY', 'SUMMER', 'WINTER', 'NULL_PRICE', 'LITINIT', 'T', 'S', 'G', 'S*G', 'T*G',
                   'T*S', 'T*S*G', 'TREND', 'DATE2'] + [x for x in df.columns if x.startswith('D_DAY')]:
        df[col] = np.log(df[col] + 1)
df['P'] = np.log(df['P'] + 1)
df['P'] = df['P'] - df['AR1'] - (df['AR1'] - df['AR2'])
print(df.shape)

df = df[exog_vars + ['P']].dropna(axis=0)

predictq = sm.OLS(endog=df['Q'], exog=df[['WORKDAY', 'TME', 'PP', 'WINTER']]).fit()
predictq = predictq.predict(df[['WORKDAY', 'TME', 'PP', 'WINTER']])
predictq = pd.DataFrame(predictq, columns=['Q_IV'])
df = pd.concat([df.drop(['Q', 'WORKDAY', 'TME', 'PP', 'WINTER'], axis=1), predictq], axis=1)

# df = sm.add_constant(df)
mod = sm.OLS(endog=df['P'], exog=df.drop(['P', 'DATE2'], axis=1))
re_res = mod.fit(cov_type='HC1')
print(re_res.summary())
'''

vif = pd.DataFrame()
drop_var = ['P', 'AR1', 'AR2', 'AR3', 'DATE2']
vif['vif'] = [variance_inflation_factor(df.drop(drop_var, axis=1).values, i) for i in
              range(df.drop(drop_var, axis=1).shape[1])]
vif['features'] = df.drop(drop_var, axis=1).columns
print(vif)

# EL PAPER DE BERTRAND ES EL UNICO QUE VEO QUE COMPARA CORR SERIAL EN DIF-IN-DIF: USAR RANDOMIZATION INFERENCE

# Parallet trend test
print(df.columns)

# Lags and leads
i = 0
subasta = ['20/03/2013', '25/06/2013', '24/12/2013']
df['DATE2'] = pd.to_datetime(df['DATE2'], format='%d/%m/%Y')
df = df[df['DATE2'] < '01/01/2014']

for col in ['trend_3_2013', 'trend_2_2013', 'trend_1_2013']:
    # Lags La interacción del tiempo y el grupo Tratado
    df['LA_' + col] = pd.Series(np.NaN, index=df.index)
    subasta_i = subasta[i]
    print(pd.to_datetime(subasta_i, format='%d/%m/%Y') - timedelta(days=dummy))
    df.loc[df['DATE2'] == (pd.to_datetime(subasta_i, format='%d/%m/%Y') - timedelta(days=dummy)), 'LA_' + col] = 1
    df['LA_' + col] = df['LA_' + col].bfill(axis=0, limit=dummy)
    df['LA_' + col] = df['LA_' + col].fillna(0)
    df['LA_' + col] = df['LA_' + col] * df['T']
    # Lead
    df['LE_' + col] = pd.Series(np.NaN, index=df.index)
    subasta_i = subasta[i]
    df.loc[df['DATE2'] == (pd.to_datetime(subasta_i, format='%d/%m/%Y')), 'LE_' + col] = 1
    df['LE_' + col] = df['LE_' + col].bfill(axis=0, limit=dummy)
    df['LE_' + col] = df['LE_' + col].fillna(0)
    df['LE_' + col] = df['LE_' + col] * df['T']
    i += 1
df.to_csv('test.csv', sep=';', index=False)
print(df.columns)
mod = sm.OLS(endog=df['P'], exog=df.drop(['P', 'DATE2', 'T*S*G', 'G', 'S*G', 'T*G', 'T*S','TREND', 'LA_trend_3_2013'], axis=1))
re_res = mod.fit(cov_type='HC1')
print(re_res.summary())

# DIF iN DIF

df = pd.read_csv(STRING.file_panel_data, sep=';')
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')
df['WEEKDAYS'] = df['DATE'].dt.dayofweek
df['MONTHS'] = df['DATE'].dt.month
dummy_day = pd.get_dummies(df['WEEKDAYS'], prefix='D_DAY', drop_first=True)
dummy_month = pd.get_dummies(df['MONTHS'], prefix='D_MONTH', drop_first=True)
df = pd.concat([df, dummy_day, dummy_month], axis=1)
df['DATE2'] = df['DATE']
df_1 = df.set_index(['DATE2', 'MARKET'])

df_1['T'] = df_1['FE_SPAIN']
df_1['S'] = df_1['DUMMY_' + back + str(dummy) + '_DAY']

df_1['T*S'] = df_1['T'] * df_1['S']

exog_vars = ['Q', 'TME', 'PP', 'PRICE_OIL', 'PRICE_GAS', 'AR1', 'AR2', 'AR3',
             'WINTER', 'NULL_PRICE', 'LITINIT', 'TREND', 'T', 'S',
             'T*S', 'WORKDAY', 'DATE'] + [x for x in df_1.columns if x.startswith('D_DAY')]
df_1['TME'] = df_1['TME'] + 39
for col in exog_vars:
    if col not in ['WORKDAY', 'SUMMER', 'WINTER', 'NULL_PRICE', 'LITINIT', 'T', 'S', 'G', 'S*G', 'T*G',
                   'T*S', 'T*S*G', 'TREND', 'DATE'] + [x for x in df_1.columns if x.startswith('D_DAY')]:
        df_1[col] = np.log(df_1[col] + 1)
df_1['P'] = np.log(df_1['P'] + 1)
df_1['P'] = df_1['P'] - df_1['AR1'] - (df_1['AR1'] - df_1['AR2'])
df_1 = df_1[exog_vars + ['P']].dropna(axis=0)


predictq = sm.OLS(endog=df_1['Q'], exog=df_1[['WORKDAY', 'TME', 'PP', 'WINTER']]).fit()
predictq = predictq.predict(df_1[['WORKDAY', 'TME', 'PP', 'WINTER']])
predictq = pd.DataFrame(predictq, columns=['Q_IV'])
df_1 = pd.concat([df_1.drop(['Q', 'WORKDAY', 'TME', 'PP', 'WINTER'], axis=1), predictq], axis=1)

df_true = df_1[df_1['DATE'] < '2014-01-01']
mod = sm.OLS(endog=df_true['P'], exog=df_true.drop(['P', 'DATE'], axis=1))
re_res = mod.fit(cov_type='HC1')
print(re_res.summary())

# Falsification Test
df_falstest = df_1[df_1['DATE'] >= '2014-01-01']
mod = sm.OLS(endog=df_falstest['P'], exog=df_falstest.drop(['P', 'DATE', 'NULL_PRICE'], axis=1))
re_res = mod.fit(cov_type='HC1')
print(re_res.summary())
'''