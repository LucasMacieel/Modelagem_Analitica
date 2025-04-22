def modify_dataframe(dataframe):
    dataframe = dataframe.melt(id_vars='ANO', var_name='MONTH', value_name='VALUE')
    month_map = {
        'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04',
        'MAI': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
        'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'
    }
    dataframe['MONTH'] = dataframe['MONTH'].map(month_map)
    dataframe['YEAR'] = dataframe['ANO'].astype(str) + '-' + dataframe['MONTH']
    dataframe = dataframe[['YEAR', 'VALUE']].sort_values(by='YEAR').reset_index(drop=True)

    return dataframe

if __name__ == '__main__':
    import pmdarima as pm
    import pandas as pd, numpy as np, seaborn as sns, matplotlib.pylab as plt
    from datetime import datetime
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Melt the dataframe to long format with 'DATE' and 'VALUE' columns
    df = pd.read_excel("PASSO_REAL.xlsx")

    df = modify_dataframe(df)

    print(df)