import pandas as pd
import STRING
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from linearmodels.iv import IV2SLS as iv2reg
sns.set()

df = pd.read_csv(STRING.file_detrend, sep=';')

df = df.set_index('DATE')

df['C'] = pd.Series(1, index=df.index)

df['LQ1'] = df['QDIF'].shift(1)
df['LQ2'] = df['QDIF'].shift(2)


# AR Component
df['ar.L1.PSPAIN'] = df['PSPAIN'].shift(1)
df['ar.L2.PSPAIN'] = df['PSPAIN'].shift(2)
df['ar.L3.PSPAIN'] = df['PSPAIN'].shift(3)
df['ar.L4.PSPAIN'] = df['PSPAIN'].shift(4)
df['ar.L5.PSPAIN'] = df['PSPAIN'].shift(5)
df['ar.L6.PSPAIN'] = df['PSPAIN'].shift(6)
df['ar.L7.PSPAIN'] = df['PSPAIN'].shift(7)

# MA component
'''
width = 1
lag = df['PSPAIN'].shift(width)
window = lag.rolling(window=width + 2)
means = pd.DataFrame(window.mean().values, columns=['ma.L1.PSPAIN'], index=df.index)
df = pd.concat([df, means], axis=1)
'''

# X, Y
y = df[['PSPAIN']]
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

instruments = ['WORKDAY', 'INDEX', 'Portugal', 'TME', 'WINTER', 'PRCP', 'TMIN', 'SUMMER', 'Norway', 'Denmark', 'Finland', 'Sweden']
inst_square = []
for inst in instruments:
    df[inst] = np.log(df[inst])
    df[inst] = (df[inst] - df[inst].shift(1)) - (df[inst].shift(1) - df[inst].shift(2))
instruments = ['WORKDAY']

df = df.dropna(axis=0)
print(df.shape)
variable_used = ['D_sum(TOTAL_IMPORTACION_ES)',
            'D_NULL_PRICE', 'D_DUMMY_BACK_60_DAY']

variable_instrumented = ['D_sum(TOTAL_PRODUCCION_ES)']

"""
To conduct IV estimations, we need to have instrumental
variables (or instruments in short) that are (R1) uncorrelated with u but (R2) partially
and sufficiently strongly correlated with y2 once the other independent variables are
controlled for.
In practice, we can test the second requirement (b), but we can not test the first
requirement (a) because u is unobservable.

To test the second requirement (b), we need
to express a reduced form equation of y2 with all of exogenous variables. Exogenous
variables include all of independent variables that are not correlated with the error term
and the instrumental variable, z.
For the instrumental variable to satisfy the second requirement (R2), the estimated
coefficient of z must be significant. 
"""

# MODEL
# First Stage: Reduced Form
# QDIF
first_reg1 = sm.OLS(endog=df[variable_instrumented], exog=df[variable_used + instruments])
results = first_reg1.fit()
print(results.summary())

"""
In the first stage
regression, we should conduct a F-test on all instruments to see if instruments are jointly
significant in the endogenous variable, y2. 
"""
ftest = sm.OLS(endog=df[variable_instrumented], exog=df[instruments])
results = ftest.fit()

print('F-Statistic', results.fvalue)
print('pvalue', results.f_pvalue)
print(results.summary())

# MODEL
# First Stage: Reduced Form
# QNORD
instruments = ['TMIN', 'INDEX']
variable_instrumented = ['D_sum(QNORD)']
first_reg1 = sm.OLS(endog=df[variable_instrumented], exog=df[variable_used + instruments])
results = first_reg1.fit()
print(results.summary())

"""
In the first stage
regression, we should conduct a F-test on all instruments to see if instruments are jointly
significant in the endogenous variable, y2. 
"""
ftest = sm.OLS(endog=df[variable_instrumented], exog=df[instruments])
results = ftest.fit()

print('F-Statistic', results.fvalue)
print('pvalue', results.f_pvalue)
print(results.summary())

# MODEL
# First Stage: Reduced Form
# QPortugal
instruments = ['Portugal']
variable_instrumented = ['D_sum(TOTAL_PRODUCCION_POR)']
first_reg1 = sm.OLS(endog=df[variable_instrumented], exog=df[variable_used + instruments])
results = first_reg1.fit()
print(results.summary())

"""
In the first stage
regression, we should conduct a F-test on all instruments to see if instruments are jointly
significant in the endogenous variable, y2. 
"""
ftest = sm.OLS(endog=df[variable_instrumented], exog=df[instruments])
results = ftest.fit()

print('F-Statistic', results.fvalue)
print('pvalue', results.f_pvalue)
print(results.summary())

# Second Stage
y = df[['PSPAIN']]
variable_used = ['D_sum(TOTAL_IMPORTACION_ES)',
            'D_NULL_PRICE', 'LQ1', 'D_sum(TOTAL_PRODUCCION_POR)', 'D_DUMMY_BACK_60_DAY', 'ar.L1.PSPAIN',
                 'ar.L2.PSPAIN', 'ar.L3.PSPAIN',
                 'ar.L4.PSPAIN', 'ar.L5.PSPAIN', 'ar.L6.PSPAIN', 'ar.L7.PSPAIN']

variable_instrumented = ['D_sum(TOTAL_PRODUCCION_ES)', 'D_sum(QNORD)']
instruments = ['WORKDAY', 'INDEX', 'Portugal', 'D_sum(TOTAL_PRODUCCION_POR)']

# QSPAIN predict
predict_qesp = sm.OLS(endog=df['D_sum(TOTAL_PRODUCCION_ES)'], exog=df[['D_sum(TOTAL_IMPORTACION_ES)',
            'D_NULL_PRICE', 'D_DUMMY_BACK_60_DAY', 'WORKDAY']]).fit()

predict_qesp = predict_qesp.predict(df[['D_sum(TOTAL_IMPORTACION_ES)',
            'D_NULL_PRICE', 'D_DUMMY_BACK_60_DAY', 'WORKDAY']])

predict_qesp = pd.DataFrame(predict_qesp, columns=['QSPAIN'])
predict_qesp['LQ1'] = predict_qesp['QSPAIN'].shift(1)

# QNORD Predict
predict_nor = sm.OLS(endog=df['D_sum(QNORD)'], exog=df[['TMIN', 'INDEX', 'D_sum(TOTAL_IMPORTACION_ES)',
            'D_NULL_PRICE', 'D_DUMMY_BACK_60_DAY']]).fit()

predict_nor = predict_nor.predict(df[['TMIN', 'INDEX', 'D_sum(TOTAL_IMPORTACION_ES)',
            'D_NULL_PRICE', 'D_DUMMY_BACK_60_DAY']])

predict_nor = pd.DataFrame(predict_nor, columns=['QNORD'])

df = pd.concat([df.drop('LQ1', axis=1), predict_nor, predict_qesp], axis=1)

mod = sm.tsa.ARMA(endog=df['PSPAIN'], exog=df[['D_sum(TOTAL_IMPORTACION_ES)',
            'D_NULL_PRICE', 'LQ1','D_DUMMY_BACK_60_DAY', 'QSPAIN', 'QNORD', 'D_sum(TOTAL_PRODUCCION_POR)']],
                  order=(7, 0), missing='drop')
results = mod.fit(trend='nc')
print(results.summary())

