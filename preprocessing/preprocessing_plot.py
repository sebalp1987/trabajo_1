import pandas as pd
import STRING
import seaborn as sns
import matplotlib.pyplot as plot
import numpy as np
sns.set()

df = pd.read_csv(STRING.final_file_hr, sep=';')
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
df = df.sort_values(by='DATE', ascending=True).reset_index(drop=True)
df['DATE'] = df['DATE'].apply(lambda x: x.date()).map(str)
x = df['DATE'].tolist()
plot.plot(df['DATE'], df['PSPAIN'], label='Spain')
plot.plot(df['PNORD'], label='Nord Pool')
xposition = ['2013-03-20', '2013-06-25', '2013-12-24']
for xv in xposition:
    plot.axvline(x=xv, color='k', linestyle='--')
plot.xticks(x[::20], fontsize=10, rotation=45)
plot.legend()
plot.show()


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
