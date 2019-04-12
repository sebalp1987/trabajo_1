import pandas as pd
import STRING
from linearmodels import RandomEffects
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

df = pd.read_csv(STRING.file_panel_data, sep=';')
df['DATE'] = pd.to_datetime(df['DATE'])
df['WEEKDAYS'] = df['DATE'].dt.dayofweek
df['MONTHS'] = df['DATE'].dt.month
print(df['MONTHS'].unique())
dummy_day = pd.get_dummies(df['WEEKDAYS'], prefix='D_DAY', drop_first=True)
dummy_month = pd.get_dummies(df['MONTHS'], prefix='D_MONTH', drop_first=True)
df = pd.concat([df, dummy_day, dummy_month], axis=1)
print(df.columns)

df = df.set_index(['MARKET', 'YEAR', 'DATE_DDD'])
df['T'] = df['DUMMY_BACK_60_DAY']
df['S'] = df['FE_SPAIN']
df['G'] = df['TIMEEF_2013']

df['S*G'] = df['S'] * df['G']
df['T*G'] = df['T'] * df['G']
df['T*S'] = df['T'] * df['S']

df['T*S*G'] = df['T'] * df['S'] * df['G']

exog_vars = ['Q', 'TME', 'PP', 'PRICE_OIL', 'PRICE_GAS', 'AR1', 'AR2', 'AR3',
             'WINTER', 'NULL_PRICE', 'LITINIT', 'TREND', 'T', 'S', 'G', 'S*G', 'T*G',
             'T*S', 'T*S*G', 'WORKDAY'] + [x for x in df.columns if x.startswith('D_DAY')]
for col in exog_vars:
    if col not in ['WORKDAY', 'SUMMER', 'WINTER', 'NULL_PRICE', 'LITINIT', 'T', 'S', 'G', 'S*G', 'T*G',
                   'T*S', 'T*S*G', 'TREND'] + [x for x in df.columns if x.startswith('D_DAY')]:
        df[col] = np.log(df[col] + 1)
df['P'] = np.log(df['P'] + 1)
df['P'] = df['P'] - df['AR1'] - (df['AR1'] - df['AR2'])
print(df.shape)
df = df[exog_vars + ['P']].dropna(axis=0)
print(df.shape)


predictq = sm.OLS(endog=df['Q'], exog=df[['WORKDAY', 'TME', 'PP', 'WINTER']]).fit()
predictq = predictq.predict(df[['WORKDAY', 'TME', 'PP', 'WINTER']])
predictq = pd.DataFrame(predictq, columns=['Q_IV'])
df = pd.concat([df.drop(['Q', 'WORKDAY', 'TME', 'PP', 'WINTER'], axis=1), predictq], axis=1)

# df = sm.add_constant(df)
mod = sm.OLS(endog=df['P'], exog=df.drop('P', axis=1))
re_res = mod.fit(cov_type='HC1')
print(re_res.summary())


vif = pd.DataFrame()
drop_var = ['P', 'AR1', 'AR2', 'AR3']
vif['vif'] = [variance_inflation_factor(df.drop(drop_var, axis=1).values, i) for i in
              range(df.drop(drop_var, axis=1).shape[1])]
vif['features'] = df.drop(drop_var, axis=1).columns
print(vif)

# EL PAPER DE BERTRAND ES EL UNICO QUE VEO QUE COMPARA CORR SERIAL EN DIF-IN-DIF: USAR RANDOMIZATION INFERENCE