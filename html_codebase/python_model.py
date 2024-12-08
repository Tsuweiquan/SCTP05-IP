import pandas as pd
import requests
import numpy as np
import math
import datetime as dt

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score  
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import seaborn as sns 
import matplotlib.pyplot as plt 
from colorama import Fore
from xgboost import XGBRegressor

# This import is to display plotly on html page #
from jinja2 import Template
from bs4 import BeautifulSoup
# This import is to display plotly on html page #

response = requests.get("https://api.twelvedata.com/time_series?apikey=da7cc643495745a78c99c491e1d4d0a6&interval=4h&symbol=GBP/USD&start_date=2020-01-01 16:23:00&end_date=2024-11-26 16:00:00&format=JSON&timezone=utc")

# You can create a dataframe with json file of url import
df = pd.DataFrame(response.json()["values"])
# df
# Convert specified columns to float
df['open'] = df['open'].astype(float)
df['high'] = df['high'].astype(float)
df['low'] = df['low'].astype(float)
df['close'] = df['close'].astype(float)

#highlighting the datetime and closing price
df.head().style.set_properties(subset=['datetime','close'], **{'background-color': 'skyblue'})
# Ensure datetime column is parsed correctly
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# Filter data for the last year
last1year_df = df[df['datetime'] > '2023-09-01']

# Select only numeric columns
numeric_df = last1year_df.select_dtypes(include=['number'])

# Compute mean and covariance
df_mean = numeric_df.mean()
cov_returns = numeric_df.cov()

# Output results
print(df_mean, cov_returns, sep='\n')
closedf = df[['datetime','close']]
print("Shape of close dataframe:", closedf.shape)
closedf = closedf[(closedf['datetime'] >= '2024-01-01') & (closedf['datetime'] <= '2024-07-26')]


close_stock = closedf.copy()
print("Total data for prediction: ",closedf.shape[0])
del closedf['datetime']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
training_size=int(len(closedf)*0.70)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=3):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

my_model = XGBRegressor(n_estimators=1000)
my_model.fit(X_train, y_train, verbose=False)
predictions = my_model.predict(X_test)
print("Mean Absolute Error - MAE : " + str(mean_absolute_error(y_test, predictions)))
print("Root Mean squared Error - RMSE : " + str(math.sqrt(mean_squared_error(y_test, predictions))))
train_predict=my_model.predict(X_train)
test_predict=my_model.predict(X_test)

train_predict = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)

# Transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])
plotdf = pd.DataFrame({'datetime': close_stock['datetime'],
                       'original_close': close_stock['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['datetime'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Close price','datetime': 'Date'})
fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 10
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        
        yhat = my_model.predict(x_input)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat.tolist())
        temp_input=temp_input[1:]
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        yhat = my_model.predict(x_input)
        
        temp_input.extend(yhat.tolist())
        lst_output.extend(yhat.tolist())
        
        i=i+1

print("Output of predicted next days: ", len(lst_output))
        
last_days=np.arange(0,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
print(last_days)
print(day_pred)
temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

names = cycle(['Last 15 days close price','Predicted next 10 days close price'])
fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Close price','index': 'Timestamp'})
fig.update_layout(title_text='Compare last 15 bars vs next 10 bars',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
# fig.show()

### Add figure to html chart ###
figure_data = fig.to_json()
html_file_path = 'index.html'
with open(html_file_path, 'r', encoding='utf-8') as file:
    content = file.read()
    
# if the html was already updated with a chart, plotly-data would be found
# this will remove the plotly-data div
if content.find('plotly-data') != -1:
    print ("reset previous chart")
    soup = BeautifulSoup(content, 'html.parser')
    div_to_replace = soup.find('div', id='plotly-figure')
    
    if div_to_replace:
        new_div = soup.new_tag('div', id='plotly-figure')
        new_div.string = '{{ fig }}'
        print (new_div)
        div_to_replace.replace_with(new_div)
    # Write back to the same HTML file
    with open(html_file_path, 'w', encoding='utf-8') as writefile:
        writefile.write(soup.prettify())
    
content_with_figure = content.replace('{{ fig }}', f'<script id="plotly-data" type="application/json">{figure_data}</script>')
# # # Write back to the same HTML file
with open(html_file_path, 'w', encoding='utf-8') as file:
    file.write(content_with_figure)
print("Figure has been successfully appended to index.html")