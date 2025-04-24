import pmdarima as pm
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict

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
    plt.close()

    return dataframe

def get_year_value_box_plot(dataframe):
    dataframe['YEAR'] = dataframe['DATE'].dt.year
    fig, ax = plt.subplots()
    fig.set_size_inches((45, 10))
    sns.boxplot(x='YEAR', y='VALUE', data=dataframe, ax=ax)
    plt.close()

    return dataframe

def get_month_value_box_plot(dataframe):
    dataframe['MONTH'] = dataframe['DATE'].dt.month
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 4))
    sns.boxplot(x='MONTH', y='VALUE', data=dataframe, ax=ax)
    plt.close()

    return dataframe

def forecast_accuracy(fcast, atual):
    mape = np.mean(np.abs(fcast - atual)/np.abs(atual)) # MAPE
    me = np.mean(fcast - atual) # ME
    mae = np.mean(np.abs(fcast - atual)) # MAE
    mpe = np.mean((fcast - atual)/atual) # MPE
    rmse = np.mean((fcast - atual)**2)**.5 # RMSE
    corr = np.corrcoef(fcast, atual)[0,1] # corr
    mins = np.amin(np.hstack([fcast[:,None],
    atual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([fcast[:,None],
    atual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs) # minmax
    acf1 = acf(fc-test)[1] # ACF1
    return({'mape':mape, 'me':me, 'mae': mae,
    'mpe': mpe, 'rmse':rmse, 'acf1':acf1,
    'corr':corr, 'minmax':minmax})

if __name__ == '__main__':
    database = pd.read_excel("PASSO_REAL.xlsx")

    database = modify_dataframe(database)

    df = database

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
    plt.close()

    result_add = seasonal_decompose(x=df['VALUE'], model='additive',
                                    extrapolate_trend = 'freq', period = int(len(df) / 2))
    plt.rcParams.update({'figure.figsize': (5, 5)})
    result_add.plot()
    plt.close()

    x = pd.plotting.autocorrelation_plot(df)
    x.plot()
    plt.close()

    ts_log = np.log(df['VALUE'])
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    plt.plot(ts_log_diff)
    plt.close()

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
    plt.close()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(df.VALUE)
    ax1.set_title('Original Series')
    ax1.axes.xaxis.set_visible(False)
    # 1st Differencing
    ax2.plot(df.VALUE.diff())
    ax2.set_title('1st Order Differencing')
    ax2.axes.xaxis.set_visible(False)
    # 2nd Differencing
    ax3.plot(df.VALUE.diff().diff())
    ax3.set_title('2nd Order Differencing')
    plt.close()

    # Not working
    # Original Series
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 10))
    axes[0].plot(df.VALUE)
    axes[0].set_title('Original Series')
    result = adfuller(df.VALUE.dropna())
    print("p-value:", result[1])
    plot_acf(df.VALUE)
    # 1st Differencing
    axes[1].plot(df.VALUE.diff())
    axes[1].set_title('1st Order Differencing')
    result = adfuller(df.VALUE.diff().dropna())
    plot_acf(database.VALUE.diff().dropna())
    print("p-value:", result[1])
    # 2nd Differencing
    axes[2].plot(df.VALUE.diff().diff())
    axes[2].set_title('2nd Order Differencing')
    plot_acf(database.VALUE.diff().diff().dropna())
    result = adfuller(df.VALUE.diff().diff().dropna())
    print("p-value:", result[1])
    plt.savefig("grafico10.png")
    plt.close()

    # d
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(15, 10))
    axes[0].plot(df.VALUE.diff())
    axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0, 5))
    plot_pacf(df.VALUE.diff().dropna())
    plt.savefig("grafico11.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(15, 10))
    axes[0].plot(df.VALUE.diff())
    axes[0].set_title('2st Differencing')
    axes[1].set(ylim=(0, 5))
    plot_pacf(df.VALUE.diff().diff().dropna())
    plt.savefig("grafico12.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(15, 10))
    axes[0].plot(df.VALUE.diff())
    axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0, 1.2))
    plot_acf(df.VALUE.diff().dropna())
    plt.savefig("grafico13.png")
    plt.close()

    model = ARIMA(df.VALUE, order=(1, 1, 2))
    model_fit = model.fit()
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plot_predict(model_fit, dynamic=False)

    # Create Training and Test
    train = database.VALUE[:900]
    test = database.VALUE[900:]
    model = ARIMA(train, order=(1, 1, 2))
    fitted = model.fit()
    # Forecast
    fc = fitted.forecast(132, alpha=0.05, h=30)  # 95% conf
    conf = fitted.conf_int()
    se = fitted.bse
    # fc, se, conf = fitted.forecast(24, alpha=0.05) # 95% conf
    # Make as pandas series
    fc_series = pd.Series(fc.values, index=test.index)
    lower_series = pd.Series(conf[0], index=test.index)
    upper_series = pd.Series(conf[1], index=test.index)
    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha = .15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)

    model1 = ARIMA(train, order=(1, 1, 2))
    fitted = model1.fit()
    # Forecast
    fc = fitted.forecast(132, alpha=0.05)  # 95% conf
    conf = fitted.conf_int()
    se = fitted.bse
    # fc, se, conf = fitted.forecast(24, alpha=0.05) # 95% conf
    # Make as pandas series
    fc_series = pd.Series(fc.values, index=test.index)
    lower_series = pd.Series(conf[0], index=test.index)
    upper_series = pd.Series(conf[1], index=test.index)
    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha = .15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)

    #Not working
    #forecast_accuracy(fc, test.values)

    model = pm.auto_arima(database.VALUE, start_p=1, start_q=1,
                          test='adf',
                          max_p=3, max_q=3, m=12,
                          start_P=0, seasonal=False,
                          d=None, D=0, trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    model.plot_diagnostics(figsize=(7, 5))
    plt.savefig("grafico18.png")
    plt.close()