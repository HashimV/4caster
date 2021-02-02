#!/usr/bin/env python
# coding: utf-8

# In[41]:


#imports
import pandas as pd
import numpy as np
import math

#import fbprophet, the time series model that
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric

#imports for the visualizations
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#reads the table from Excel and puts in dataframe
df=pd.read_excel(r'/Users/hashimvekar/Documents/Data Science Assignment.xls')

#drop duplicates
df=df.dropna()

#converts week column to datetime
df['week']=pd.to_datetime(df['week'])

#add various columns to the data frame
df['revenue']=df.orders*df.price

#can add any other calculations that are necessary


#Sums orders, views, cart_adds, price, and inventory
totalDF=df.groupby("product").sum()

#can't sum up price in this regard, doesn't make sense
del totalDF['price']

totalDF=totalDF.round(0)

#important metrics to consider
totalDF['orders as % of cart_adds']=round(totalDF.orders/totalDF.cart_adds,2)
totalDF['orders as % of inventory']=round(totalDF.orders/totalDF.inventory,2)

#print(totalDF)

#averages on a weekly basis
averageDF=df.groupby("product").mean()

averageDF=averageDF.round(0)

averageDF['orders as % of cart_adds']=round(averageDF.orders/averageDF.cart_adds,2)
averageDF['orders as % of inventory']=round(averageDF.orders/averageDF.inventory,2)

#print(averageDF)

#determining number of rows and columns to see if they match
print(df.shape[0], 'rows', 'AND', df.shape[1],'columns')

#seeing the earliest date and latest date of the data
print('Min Date:',df.week.min(),'AND','Max Date:',df.week.max())

#learn the types of data in the table
df.head()
df.info()

#have to model one product at a time

productDF=df.loc[df['product']=="C"]

#plot time series data

plt.plot(productDF['week'],productDF['orders'])
plt.plot(productDF['week'],productDF['orders'],'k.')
plt.plot(style='k')
plt.xlabel("Date")
plt.ylabel("Number of Orders")

#Modeling

#futureDF=productDF.tail(10)

#the number of weeks you want to forecast
periods=8

#the actual data points for the periods you want to forecast
actualDF=productDF.tail(periods)

#dropping the last periods of data from the data frame

#
#There are around 50 data points per product. Therefore, the model is not as accurate with this data set.
#I am positive the model will function well when there are thousands of products and many data points.

#productDF.drop(productDF.tail(8).index,inplace=True) #drop 10 rows
#productDF.reset_index()


#IMPLEMENTING THE MODEL

#includes yearly seasonality, weekly seasonality, principles of fourier 

#with yearly seasonality, 95% confidence intervals
model1=Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)
model1.add_seasonality(name='weekly', period=10, fourier_order=3, prior_scale=0.02)

#has the functionality to incorporate holidays where demand can change

#wasn't able to implement it due to time constraints
#model2=Prophet(holidays=holidays)
#model2.add_country_holidays(country_name='Canada')

#rename the columns in the format the model wants
productDF=productDF.rename(columns={"week": "ds", "orders":"y"})

#fitting the model
model1.fit(productDF)

#making a future dataframe - taking the amount of weeks you want to forecast for
future=model1.make_future_dataframe(periods, freq="w")
future.tail()

#print(future)

#predicing the value of the forecast
forecast=model1.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']]

#print(forecast)

#plot the forecast alongside the actual dataframe
plt.plot(forecast['ds'].tail(periods),forecast['yhat'].tail(periods),'k.', color='b')
plt.plot(actualDF['week'],actualDF['orders'],'k.')

fig1=model1.plot(forecast)
fig2=model1.plot_components(forecast)

#cross validation

#need to specify the initial amount of data, the cut off period, and the horizon
productDF_cv = cross_validation(model1,initial="300 days", period="300 days", horizon="14 days")
productDF_cv.head()

#print(productDF_cv)

#calculate the performance metrics such as mse, rmse, mae, mape, mdape, coverage
productDF_p = performance_metrics(productDF_cv)
productDF_p.head()

#print(productDF_p)

#graph the cross validation metrics
graph = plot_cross_validation_metric(productDF_cv, metric='mse')

graph2 = plot_cross_validation_metric(productDF_cv, metric='rmse')

graph3 = plot_cross_validation_metric(productDF_cv, metric='mae')

graph4 = plot_cross_validation_metric(productDF_cv, metric='mape')

graph5 = plot_cross_validation_metric(productDF_cv, metric='mdape')

graph6 = plot_cross_validation_metric(productDF_cv, metric='coverage')




# 

# In[40]:





# In[27]:





# In[ ]:




