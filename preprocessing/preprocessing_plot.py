import pandas as pd
import STRING
import seaborn as sns
import matplotlib.pyplot as plot
import numpy as np
import statsmodels.api as sm
from datetime import timedelta
sns.set()

df = pd.read_csv(STRING.final_file_hr, sep=';')
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
df = df[df['DATE'] <= '2013-12-31']
df = df.sort_values(by='DATE', ascending=True).reset_index(drop=True)
df['DATE'] = df['DATE'].apply(lambda x: x.date()).map(str)
wft_ciclo, wft_tend = sm.tsa.filters.hpfilter(df['PSPAIN'])
df['PSPAIN'] = wft_tend
'''
plot.plot(df['DATE'], df['PSPAIN'], label='Spain')
x = df['DATE'].tolist()
plot.xticks(x[::90], fontsize=10, rotation=45)
plot.ylabel('Mean Average Price (€)')
plot.show()
plot.close()
'''

wft_ciclo, wft_tend = sm.tsa.filters.hpfilter(df['PNORD'])
df['PNORD'] = wft_tend
position = ['2013-03-20', '2013-06-25', '2013-12-24']
for i in position:
    df_i = df[df['DATE'] <= i]

    if position.index(i) > 0:
        df_i = df_i[df_i['DATE'] >= position[position.index(i) - 1]]

    plot.plot(df_i['DATE'], df_i['PSPAIN'], label='Spain')
    plot.plot(df_i['DATE'], df_i['PNORD'], label='Nord Pool')
    xv = pd.to_datetime(i, format='%Y-%m-%d') - timedelta(days=70)
    x = df_i['DATE'].tolist()
    plot.axvline(x=xv.strftime('%Y-%m-%d'), color='k', linestyle='--', labeL='70 days before')
    plot.axvline(x=i, color='k', linestyle='-', label='Auction')
    plot.xticks(x[::5], fontsize=10, rotation=45)
    plot.title('Parallel Trends')
    plot.legend(loc='lower right')
    plot.show()
    plot.close()


# Weather condition
plot.plot(df['DATE'], df['TME_BCN'], label='Tavg (Cº)')
plot.plot(df['TMAX_BCN'], label='Tmax (Cº)')
plot.plot(df['TMIN_BCN'], label='Tmin (Cº)')
x = df['DATE'].tolist()
plot.xticks(x[::20], fontsize=10, rotation=45)
plot.ylabel('Temperature (Cº)')
plot.legend(loc="lower right")
plot.show()

# Weather condition
plot.plot(df['DATE'], df['TAVG'], label='Tavg (Cº)')
plot.plot(df['TMAX'], label='Tmax (Cº)')
plot.plot(df['TMIN'], label='Tmin (Cº)')
x = df['DATE'].tolist()
plot.xticks(x[::20], fontsize=10, rotation=45)
plot.ylabel('Temperature (Cº)')
plot.legend(loc="lower right")
plot.show()


# Demand
df['QDIF'] = df['sum(TOTAL_PRODUCCION_ES)'] - df['sum(QNORD)']
plot.plot(df['DATE'], df['QDIF'], label='QDIF')
plot.plot(df['sum(TOTAL_PRODUCCION_ES)'], label='QSPAIN')
plot.plot(df['sum(QNORD)'], label='QNORD')
x = df['DATE'].tolist()
plot.xticks(x[::20], fontsize=10, rotation=45)
plot.ylabel('MWh')
plot.legend(loc="lower right")
plot.show()
