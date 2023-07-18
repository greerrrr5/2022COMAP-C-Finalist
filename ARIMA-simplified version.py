import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r'D:\bit_1.csv',index_col=0, parse_dates=True)
df.head()
data = df['Value']
plt.figure(figsize = (10, 6))
plt.plot(df.index, data)
plt.show() 
data_diff = data.diff()
data_diff = data_diff.dropna()
plt.plot(data_diff)
plt.show()
from statsmodels.tsa.stattools import adfuller
print(adfuller(data_diff))
from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(data_diff, lags = 20)) 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
pacf = plot_pacf(data_diff, lags=20)
plt.title('PACF')
pacf.show()
acf = plot_acf(data_diff, lags=20)
plt.title('ACF')
acf.show()
import statsmodels.tsa.stattools as st
model = st.arma_order_select_ic(data_diff, max_ar=5, max_ma=5, ic=['aic', 'bic', 'hqic'])
print(model.bic_min_order) 
from statsmodels.tsa.arima_model import ARMA
model_arma = ARMA(data_diff, order = (2,2))
result_arma = model_arma.fit(disp = -1, method = 'css')
resid = result_arma.resid
from statsmodels.graphics.api import qqplot
qqplot(resid, line='q', fit=True)
plt.show()
forecast=result_arma.forecast(15)  forecast=pd.Series(forecast[0],index=pd.period_range(start=str(date[i]),end=str(date[i+14]),freq='D'))
forecast[0]=forecast[0]+data_temp[-1]
    for j in range(1,15):
        forecast[j]=forecast[j-1]+forecast[j]
forecast
