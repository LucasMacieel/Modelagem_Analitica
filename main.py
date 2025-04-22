def modify_dataframe(dataframe):
    dataframe = dataframe.melt(id_vars='ANO', var_name='MONTH', value_name='VALUE')
    month_map = {
        'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04',
        'MAI': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
        'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'
    }
    dataframe['MONTH'] = dataframe['MONTH'].map(month_map)
    dataframe['DATE'] = dataframe['ANO'].astype(str) + '-' + dataframe['MONTH']
    dataframe = dataframe[['DATE', 'VALUE']].sort_values(by='DATE').reset_index(drop=True)

    return dataframe

def get_date_value_plot(dataframe):
    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])
    dataframe.set_index('DATE', inplace=True)
    dataframe.plot(figsize=(20, 10))
    plt.savefig("grafico1.png")

    return dataframe

def get_year_value_box_plot(dataframe):
    dataframe['YEAR'] = dataframe['DATE'].dt.year
    fig, ax = plt.subplots()
    fig.set_size_inches((45, 10))
    sns.boxplot(x='YEAR', y='VALUE', data=dataframe, ax=ax)
    plt.savefig("grafico2.png")

    return dataframe

def get_month_value_box_plot(dataframe):
    dataframe['MONTH'] = dataframe['DATE'].dt.month
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 4))
    sns.boxplot(x='MONTH', y='VALUE', data=dataframe, ax=ax)
    plt.savefig("grafico3.png")

    return dataframe

if __name__ == '__main__':
    import pmdarima as pm
    import pandas as pd, numpy as np, seaborn as sns, matplotlib.pylab as plt
    from datetime import datetime
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose

    df = pd.read_excel("PASSO_REAL.xlsx")

    df = modify_dataframe(df)

    df = get_date_value_plot(df)

    df.reset_index(inplace=True)

    df = get_year_value_box_plot(df)

    df = get_month_value_box_plot(df)

    average = df['VALUE'].to_list()
    result = adfuller(average)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    df.set_index(['DATE'], inplace=True)
    result_add = seasonal_decompose(x=df['VALUE'], model='multiplicative',
                                    extrapolate_trend = 'freq', period = int(len(df) / 2))
    plt.rcParams.update({'figure.figsize': (5, 5)})
    result_add.plot()
    plt.savefig("grafico4.png")

    result_add = seasonal_decompose(x=df['VALUE'], model='additive',
                                    extrapolate_trend = 'freq', period = int(len(df) / 2))
    plt.rcParams.update({'figure.figsize': (5, 5)})
    result_add.plot()
    plt.savefig("grafico5.png")
    plt.close("all")

    x = pd.plotting.autocorrelation_plot(df)
    x.plot()
    plt.savefig("grafico6.png")
    plt.close("all")

    ts_log = np.log(df['VALUE'])
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    plt.plot(ts_log_diff)
    plt.savefig("grafico7.png")
    plt.close("all")

    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20)
    # Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')
    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.savefig("grafico8.png")