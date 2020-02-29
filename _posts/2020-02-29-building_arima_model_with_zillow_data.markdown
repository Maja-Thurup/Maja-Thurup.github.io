---
layout: post
title:      "Building Arima model with Zillow data "
date:       2020-02-29 20:17:56 +0000
permalink:  building_arima_model_with_zillow_data
---


In this project I was challenged to find top 5 zipcodes for investing in real estate. I used data from [Zillow]((https://www.zillow.com/research/data/) that contains all monthly median prices for all zipcodes in USA from year 1996 to 2018. 
To complete this project successfully I had to answer the following questions:
* What is the definition of best?
* What methods do I use to find best zipcodes?
* What method should I use to predict prices?

In this post I am going to try and answer them.
# Filtering for best Zipcodes
So first of all I filtered through all the data and found zicodes that  have beed increasing in value steadily for 20 years. I calculated mean monthly growth for each zipcode and picked the highest ones. Here is how I did it:

```
import pandas as pd

df = pd.read_csv('zillow_data.csv')
print(df.shape)
df.head()
```

![Zillow](https://i.imgur.com/hiyomLO.png)


```
zipcodes = df.drop(['RegionID', 'City', 'State', 'Metro', 'CountyName', 'SizeRank'], axis=1)
zipcodes.dropna(inplace=True)
df.dropna(inplace=True) ```


Create df to store and compare my findings:

```df_gr = pd.DataFrame(df, columns=['RegionName', 'City', 'State'])
cols = zipcodes.drop(['RegionName'], axis=1)
df_gr['growth_rate'] = 0```

Find and store the percentage change for each month:

```
for num in range(len(cols.columns)-1):
    df_gr['growth_rate'] += (1 - cols.iloc[:,num] / cols.iloc[:,num+1]) * 100
```

Divide by the total number of months and find the mean change in %   

```
df_gr['growth_rate'] = df_gr['growth_rate'] / (len(cols.columns)-1)
```

Find difference between first and last price in %:

```
df_gr['first_vs_last'] = (1 - zipcodes['1996-04'] / zipcodes['2018-04']) * 100
df_gr['current_price'] = zipcodes['2018-04']
```


Sort values in the ascending order:

```df_gr.sort_values('growth_rate', ascending=False).head(10)```

![](https://i.imgur.com/XFBw8Qm.png)

So these zipcodes have the fastest growing house prices in the country.  They must be a safe bet in a long run. Lets have a visual on these.
Learn.co provided me with some helpful functions for transforming the data:

```
def get_datetimes(df):
    
    '''transforms dates to datetime format'''
    
    return pd.to_datetime(df.columns.values[3:], format='%Y-%m')

def melt_data(df):
    
    '''transforms data to long format'''
    
    melted = pd.melt(df, id_vars=['RegionName', 'City', 'State'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value':'mean'})
```

Now lets see how prices changed over the years:

First lets drop all rows that doesn't satisfy our conditions:

```
df['growth_rate'] = df_gr['growth_rate']
df['first_vs_last'] = df_gr['first_vs_last']

df = df[df['2018-04'] >= 1000000]
df = df[(df['growth_rate'] >= 0.75) & (df['first_vs_last'] >= 85)]
```

Now to the plot itself:

```
zips = df.RegionName.values
plt.figure(figsize=(19,10))
palette = plt.get_cmap('tab20')

for ind, zipcode in enumerate(zips):
    df_melt = melt_data(df[df['RegionName'] == zipcode])
    city = df.loc[df['RegionName'] == zipcode, 'City'].values[0]
    state = df.loc[df['RegionName'] == zipcode, 'State'].values[0]
    plt.plot(df_melt, label=[zipcode, city, state], color=palette(ind), linewidth=3)   

plt.title('Price change over the years', fontsize=30)
plt.ylabel('price', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
leg = plt.legend(loc='upper left', fontsize=16)

for legobj in leg.legendHandles:
    legobj.set_linewidth(10.0)

plt.gcf().autofmt_xdate()
plt.show()
```

![](https://i.imgur.com/eUELOsN.png)

On the plot above we can see how well all these areas were doing lately. They have recoved pretty fast after the 2008 recession. All real estate in these areas have been in demand througout 20 years which means that peope will always want to live in these areas. This makes these areas a good choice for long term investment because even after recession they will eventually keep growing in prices.

# Forecasting prices for next year

One of the most commons methods used for time series modeling is ARIMA.

ARIMA, short for 'Auto Regressive Integrated Moving Average' is actually a class of models that 'explains' a given time series based on its own past values, that is, its own lags and the lagged forecast errors, so that equation can be used to forecast future values.

Lets build ARIMA model for one of the zipcodes.

First of all let's build some usefull functions so we can repeat same process for other zipcodes.

ARIMA model irequires p,d,q parameters where:

* p is the number of autoregressive terms,
* d is the number of nonseasonal differences needed for stationarity, and
* q is the number of lagged forecast errors in the prediction equation.

There are multiple ways to find best parameters for a model. One of them is The Akaike information criterion (AIC). 
It is an estimator of out-of-sample prediction error and thereby relative quality of statistical models for a given set of data. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models. Thus, AIC provides a means for model selection.
So the function below tries different parameters on AIC estimator and returns the best parameter.

```
from statsmodels.tsa.arima.model import ARIMA
import itertools

def find_pdq(ts, param_pool=5):
    
    '''returns p,d,q parameters with highest aic for ARIMA model'''
   
    p=d=q=range(0,param_pool)
    pdq = list(itertools.product(p,d,q))
    aics = []
    for param in pdq:
        try:
            model_arima = ARIMA(ts,order=param)
            model_arima_fit = model_arima.fit()
            aics.append([param, model_arima_fit.aic])
        except:
            continue
    aics_df = pd.DataFrame(aics, columns=['pdq', 'aic'])
    pdq = aics_df.loc[aics_df['aic'].idxmin()]
    return pdq[0]
```

Also  we need a function that predicts values for next 12 month and plots the prediction:

```
def forecast(ts, periods=1):
    
    '''Predicts values for given number of periods using ARIMA model.
       Returns: list of predicted values, time series with predicted values, 
             rate of change between last actual value and last predicted value '''
        
    predictions = []
    time_s = ts.copy()
    
    for t in range(periods):
        pdq = find_pdq(time_s)
        model = ARIMA(time_s, order=pdq, enforce_stationarity=False, 
                      enforce_invertibility=False)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        time_s.loc[time_s.index.max()+pd.DateOffset(months=1)] = yhat
    
    growth_rate = (1 - (ts.value[-1] / predictions[-1])) * 100
    return predictions, time_s['2018-04-01':], growth_rate 
```

```
def plot_forecast(ts, periods=1):
    
    '''Predicts values for given number of periods using ARIMA model.
       Returns: plot with actual and predicted values marked with different colors,
                rate of change between last actual value and last predicted value'''
    
    forecast_ = forecast(ts, periods=periods)
    
    print('Predicted growth rate in ', periods, 'month period = ', np.round(forecast_[2], 1), '%')
    
    plt.figure(figsize=(14,7))
    plt.plot(ts['2008-04-01':], label='prices') # plots actual data starting at year 2008 for better view
    plt.plot(forecast_[1], label='predictions', color='red') #plots predicted data in red color
    plt.axvline('2018-04-01', color='black') # show where forecast starts
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', fontsize=14)
    plt.show()
    
    return forecast_[2] # returns growth rate
```
 Now we have all we need. Let's plot forecast for one of the zipcodes.
  
```
df_melt = melt_data(df[df['RegionName'] == 11216])
 result = plot_forecast(df_melt, periods=12) # plots forecast and calculates its growth rate
```

New York NY 11216:

Predicted growth rate in  12 month period =  7.4 %
![](https://i.imgur.com/yBEkApO.png)

# Conclusion
The best area for investment is the one that brings maximum profit with minimum risks.

* First It has to have good history of prices with steady growth over the years. This will make it a safe candidate for a long term investment and lower risk of losing money in a long run.

* Second It has to show a good upward trend in recent years and in near future predictions.

A balance between these two factors makes a best area for investment.




