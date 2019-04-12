params = {

    'total_var': ['PSPAIN', 'PPORTUGAL', 'QNORD', 'DUMMY', 'sum(TOTAL_IMPORTACION_ES)',
                  'sum(TOTAL_PRODUCCION_ES)', 'sum(TOTAL_DEMANDA_NAC_ES)',
                  'sum(TOTAL_EXPORTACIONES_ES)', 'sum(TOTAL_DDA_ES)',
                  'sum(TOTAL_POT_IND_ES)', 'sum(TOTAL_PRODUCCION_POR)',
                  'sum(TOTAL_DEMANDA_POR)', 'sum(HIDRAULICA_CONVENC)',
                  'sum(HIDRAULICA_BOMBEO)', 'sum(NUCLEAR)', 'sum(CARBON NACIONAL)',
                  'sum(CARBON_IMPO)', 'sum(CICLO_COMBINADO)', 'sum(FUEL_SIN_PRIMA)',
                  'sum(FUEL_PRIMA)', 'sum(REG_ESPECIAL)', 'TREND', 'PRICE_OIL',
                  'PRICE_GAS', 'RISK_PREMIUM', 'TME_MADRID', 'TMAX_MADRID', 'TMIN_MADRID',
                  'PP_MADRID', 'TME_BCN', 'TMAX_BCN', 'TMIN_BCN', 'PP_BCN', 'GDP',
                  '%EOLICA', 'DUMMY_BACK_5_DAY',
       'DUMMY_BACK_10_DAY', 'DUMMY_BACK_15_DAY', 'DUMMY_BACK_20_DAY',
       'DUMMY_BACK_25_DAY', 'DUMMY_BACK_30_DAY', 'DUMMY_BACK_45_DAY',
       'DUMMY_FORW_5_DAY', 'DUMMY_FORW_10_DAY', 'DUMMY_FORW_15_DAY',
       'DUMMY_FORW_20_DAY', 'DUMMY_FORW_25_DAY', 'DUMMY_FORW_30_DAY',
       'DUMMY_FORW_45_DAY', 'DUMMY_2010_REGIMEN', 'WORKDAY',
                  'SUMMER', 'WINTER'],

    'linear_var1': ['PSPAIN', 'DUMMY',

                   'sum(HIDRAULICA_CONVENC)',
                   'sum(HIDRAULICA_BOMBEO)', 'sum(NUCLEAR)', 'sum(CARBON NACIONAL)',
                   'sum(CARBON_IMPO)',
                   'sum(FUEL_PRIMA)', 'sum(REG_ESPECIAL)', 'TREND',
                    'DUMMY_2010_REGIMEN',
                   'WINTER'],

    'linear_var2': ['PSPAIN', 'DUMMY', 'sum(TOTAL_IMPORTACION_ES)',

                  'sum(TOTAL_EXPORTACIONES_ES)',
                  'sum(TOTAL_POT_IND_ES)', 'sum(TOTAL_PRODUCCION_POR)',
                  'sum(HIDRAULICA_CONVENC)',
                  'sum(HIDRAULICA_BOMBEO)', 'sum(NUCLEAR)', 'sum(CARBON NACIONAL)',
                  'sum(CARBON_IMPO)', 'sum(CICLO_COMBINADO)', 'sum(FUEL_SIN_PRIMA)',
                  'sum(FUEL_PRIMA)', 'sum(REG_ESPECIAL)', 'TREND', 'PRICE_OIL',
                  'PRICE_GAS', 'RISK_PREMIUM',
                  '%EOLICA'
                    ],
    'linear_var': ['PSPAIN', 'DUMMY_BACK_50_DAY', 'sum(TOTAL_IMPORTACION_ES)',
                  'QDIF', 'sum(QNORD)', 'sum(TOTAL_PRODUCCION_ES)'

                  'sum(TOTAL_POT_IND_ES)', 'sum(TOTAL_PRODUCCION_POR)',

                  'sum(CARBON_IMPO)',  'TREND', 'PRICE_OIL',
                  'PRICE_GAS', 'RISK_PREMIUM',
                  '%EOLICA',
                  'SUMMER', 'WINTER', 'NULL_PRICE']
}
