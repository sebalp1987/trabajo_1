import STRING
import pandas as pd
import numpy as np
import sys

from pyspark.sql.functions import col, udf, regexp_replace, upper, concat, lit
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.window import Window

from resource.spark import SparkJob


class Preprocess(SparkJob):

    def __init__(self):
        self._spark = self.get_spark_session("IdTask")

    def run(self):
        df_price, df_demanda, df_produccion, df_pinternac, df_subasta = self._extract_data()
        df = self._transform_data(df_price, df_demanda, df_produccion, df_pinternac, df_subasta)
        self._load_data(df)
        self._spark.stop()

    def _extract_data(self):
        """Load data from Parquet file format.
        :return: Spark DataFrame.
        """

        df_price = (self._spark.
                    read.
                    csv(STRING.file_precio_daily, header=True, sep=';'))

        df_demanda = (self._spark.
                      read.
                      csv(STRING.file_demanda, header=True, sep=';'))

        df_produccion = (self._spark.
                         read.
                         csv(STRING.file_output, header=True, sep=';'))

        df_pinternac = pd.read_csv(STRING.file_pinternac, sep=';')

        df_subasta = self._spark.read.csv(STRING.file_subasta, header=True, sep=';')

        return df_price, df_demanda, df_produccion, df_pinternac, df_subasta

    @staticmethod
    def _transform_data(df_price, df_demanda, df_produccion, df_pinternac, df_subasta):
        """Transform original dataset.

        :param df: Input DataFrame.
        :param redes: Redes file that comes from Investigation Office
        :param entity_: Entity Zurich 'Z' or Another (BANC SABADELL 'BS')
        :return: Transformed DataFrame.
        """

        # Correct Decimals by dots
        bad_columns = ['TOTAL_IMPORTACION_ES', 'TOTAL_PRODUCCION_ES', 'TOTAL_DEMANDA_NAC_ES', 'TOTAL_EXPORTACIONES_ES',
                       'TOTAL_DDA_ES', 'TOTAL_POT_IND_ES', 'TOTAL_PRODUCCION_POR', 'TOTAL_DEMANDA_POR']
        for i in bad_columns:
            df_demanda = (
                df_demanda
                    .withColumn(i, regexp_replace(i, '\\.', ''))
                    .withColumn(i, regexp_replace(i, ',', '.').cast('float'))

            )

        bad_columns = ['HIDRAULICA_CONVENC', 'HIDRAULICA_BOMBEO', 'NUCLEAR', 'CARBON NACIONAL',
                       'CARBON_IMPO', 'CICLO_COMBINADO',
                       'FUEL_SIN_PRIMA', 'FUEL_PRIMA', 'REG_ESPECIAL']
        for i in bad_columns:
            df_produccion = (
                df_produccion
                    .withColumn(i, regexp_replace(i, '\\.', ''))
                    .withColumn(i, regexp_replace(i, ',', '.').cast('float'))
            )
        # Estos son producción cero o importación cero
        df_produccion = df_produccion.fillna(0)
        df_demanda = df_demanda.fillna(0)

        # Date Variables
        df_price = df_price.select(*['ANIO', 'MES', 'DIA', 'HORA', 'PESPANIA', 'PPORTUGAL', 'PNORD', 'QNORD'])
        funct = udf(lambda x: x.zfill(2), StringType())
        df_price = df_price.withColumn('HORA', (df_price['HORA']-1))
        df_price = df_price.withColumn('HORA', df_price['HORA'].cast(IntegerType()).cast(StringType()))
        df_price.show()
        df_produccion = df_produccion.withColumn('HORA', df_produccion['HORA']-1)
        df_produccion = df_produccion.withColumn('HORA', df_produccion['HORA'].cast(IntegerType()).cast(StringType()))
        df_demanda = df_demanda.withColumn('HORA', df_demanda['HORA']-1)
        df_demanda = df_demanda.withColumn('HORA', df_demanda['HORA'].cast(IntegerType()).cast(StringType()))
        df_price = df_price.withColumn('HORA', funct(df_price['HORA']))
        df_produccion = df_produccion.withColumn('HORA', funct(df_produccion['HORA']))
        df_demanda = df_demanda.withColumn('HORA', funct(df_demanda['HORA']))
        df_price = df_price.withColumn('DIA', funct(df_price['DIA']))
        df_produccion = df_produccion.withColumn('DIA', funct(df_produccion['DIA']))
        df_demanda = df_demanda.withColumn('DIA', funct(df_demanda['DIA']))
        df_price = df_price.withColumn('MES', funct(df_price['MES']))
        df_produccion = df_produccion.withColumn('MES', funct(df_produccion['MES']))
        df_demanda = df_demanda.withColumn('MES', funct(df_demanda['MES']))

        df_demanda = df_demanda.withColumn('DATE', concat(col('DIA'), lit('-'), col('MES'), lit('-'), col('ANIO')))
        df_produccion = df_produccion.withColumn('DATE',
                                                 concat(col('DIA'), lit('-'), col('MES'), lit('-'), col('ANIO')))
        df_price = df_price.withColumn('DATE', concat(col('DIA'), lit('-'), col('MES'), lit('-'), col('ANIO')))

        df_demanda = df_demanda.withColumn('DATE_HOUR', concat(col('DATE'), lit(' '), col('HORA'), lit(':00:00')))
        df_produccion = df_produccion.withColumn('DATE_HOUR', concat(col('DATE'), lit(' '), col('HORA'), lit(':00:00')))
        df_price = df_price.withColumn('DATE_HOUR', concat(col('DATE'), lit(' '), col('HORA'), lit(':00:00')))

        # SUBASTA
        df = df_price.join(df_subasta, how='left', on='DATE').fillna({'DUMMY': 0})
        delete_var = ['ANIO', 'MES', 'DIA', 'HORA', 'DATE']
        df_demanda = df_demanda.drop(*delete_var)
        df_produccion = df_produccion.drop(*delete_var)
        df = df.drop(*['ANIO', 'DIA', 'HORA'])

        df = df.join(df_demanda, how='left', on='DATE_HOUR')
        df = df.join(df_produccion, how='left', on='DATE_HOUR')

        # INTERPOLATE
        df.show()
        df = df.toPandas()
        df_pinternac = df_pinternac.interpolate(limit_direction='backward', method='nearest')
        df = pd.merge(df, df_pinternac, how='left', left_on='DATE', right_on='FECHA')

        del df['FECHA']

        # DUMMY VARS
        dummy_var = [5, 10, 15, 20, 30]
        df.loc[df['DUMMY'] == 0, 'DUMMY'] = np.NaN
        for i in dummy_var:
            name = 'DUMMY_' + str(i) + '_DAY'
            df[name] = pd.Series(df['DUMMY'], index=df.index)
            rows = i * 24
            df[name] = df[name].interpolate(limit=rows, limit_direction='backward', method='values')
            df[name] = df[name].fillna(0)
        print(df)
        df['DUMMY'] = df['DUMMY'].fillna(0)
        df = df.dropna(axis=0, how='any')
        print(df)

        # WORK DAY
        df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
        df['WEEKDAY'] = df['DATE'].dt.dayofweek
        df['MES'] = df['DATE'].dt.month

        df['WORKDAY'] = pd.Series(0, index=df.index)
        df.loc[df['WEEKDAY'].isin([0, 1, 2, 3, 4]), 'WORKDAY'] = 1
        del df['WEEKDAY']

        # SUMMER-WINTER
        df['SUMMER'] = pd.Series(0, index=df.index)
        df.loc[df['MES'].isin([7, 8]), 'SUMMER'] = 1
        df['WINTER'] = pd.Series(0, index=df.index)
        df.loc[df['MES'].isin([12, 1]), 'WINTER'] = 1
        del df['MES']
        bool_cols = [col for col in df
                     if df[[col]].dropna().isin([0, 1]).all().values]
        for i in df.drop(bool_cols + ['DATE', 'DATE_HOUR'], axis=1).columns.values.tolist():
            df[i] = df[i].map(float)
            df[i] = df[i].round(2)

        del df['DATE']
        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.
        :param df: DataFrame to print.
        :return: None
        """
        df.to_csv(STRING.final_file_hr, sep=';', index=False)


if __name__ == '__main__':
    Preprocess().run()
