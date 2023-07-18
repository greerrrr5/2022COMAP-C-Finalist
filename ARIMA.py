# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:00:25 2022
ARIMA attempt1
@author: 86136
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r'D:\bit_1.csv',index_col=0, parse_dates=True)
df.head()
data = df['Value']
# 第三步：画图，观察数据
plt.figure(figsize = (10, 6))
plt.plot(df.index, data)
plt.show() # 从肉眼看不符合平稳时间序列
# 第四步：做一阶差分，观察数据
data_diff = data.diff()
data_diff = data_diff.dropna()
plt.plot(data_diff)
plt.show() # 肉眼看，大致是平稳时间序列

# 第五步：单位根检验，确定数据为平稳时间序列
from statsmodels.tsa.stattools import adfuller
print('单位根检验结果：')
print(adfuller(data_diff))
# -12.574267463093953,:ADF检验的结果
# 1.9731589957016685e-23：P值
# 33：滞后数量
# 5650：用于ADF回归和临界值计算的数量
# 字典：1% 5% 10% 临界值
# 从结果可以看出拒绝原假设（原假设为是平稳时间序列），故数据为平稳时间序列

# 第六步：Q检验-检验数据是否具有相关性
# 只有在序列有相关性，即t时刻的y与t-1时刻的y有关系时arma模型才有意义
from statsmodels.stats.diagnostic import acorr_ljungbox
print('Q检验结果：')
print(acorr_ljungbox(data_diff, lags = 20)) # 第一个数：统计值； 第二个数：p值
# 从结果可以看出，p值较小，拒绝原假设（没有相关性），故数据有序列相关性
# 第七步：确定AR和MA的阶数-初步判断：画acf、pacf图（该种方式有时判断不出来）

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
pacf = plot_pacf(data_diff, lags=20)
plt.title('PACF')
pacf.show()
acf = plot_acf(data_diff, lags=20)
plt.title('ACF')
acf.show()
'''
# 第八步：使用AIC、BIC最小准则确定p、q；当p、q阶数较小时，可用这种暴力解法

import statsmodels.tsa.stattools as st
model = st.arma_order_select_ic(data_diff, max_ar=5, max_ma=5, ic=['aic', 'bic', 'hqic'])
print(model.bic_min_order) #返回一个元组，分别为p值和q值
'''
## 第九步：拟合ARIMA或者ARMA模型

# 当使用data_diff的数据时，拟合ARMA模型：

from statsmodels.tsa.arima_model import ARMA

model_arma = ARMA(data_diff, order = (2,2))

result_arma = model_arma.fit(disp = -1, method = 'css')

# 第十步：检验模型效果：残差检验
# 如果残差是白噪声序列，说明时间序列中有用的信息已经被提取完毕了，剩下的全是随机扰动，是无法预测和使用的。
# 残差序列如果通过了白噪声检验，则建模就可以终止了，因为没有信息可以继续提取。
# 如果残差如果未通过白噪声检验，说明残差中还有有用的信息，需要修改模型或者进一步提取。
resid = result_arma.resid
from statsmodels.graphics.api import qqplot
qqplot(resid, line='q', fit=True)
plt.show()
# qq图中：如果是正态分布则为一条直线，即红线。结果大致符合白噪声
# 白噪声检验除了qq图还可以使用DW检验法（DW：检验残差序列是否具有自相关性，只适用一一阶自相关；多阶自相关可用LM检验）
import statsmodels.api as sm
print(sm.stats.durbin_watson(resid.values))
# 第十一步：预测
pred =result_arma.predict(start=2, end =len(data)+11 ) # 从训练集第0个开始预测(start=1表示从第0个开始)，预测完整个训练集后，还需要向后预测10个
print(len(pred))
print(pred[-10:]) # 有负数，表明是一阶差分之后的
# 第十二步：将预测的平稳值还原为非平稳序列
result_fina = np.array(pred[0:-10])+ (np.array(data.shift(1)))
print(result_fina[-10:])

# 此时可以画出预测的曲线和data_diff进行比较
plt.plot(result_fina)
plt.plot(data)
plt.show()