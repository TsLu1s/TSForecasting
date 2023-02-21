import pandas as pd
import numpy as np
import sys
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from fbprophet import Prophet
from pmdarima.arima import auto_arima
from .tsf_data_eng import slice_timestamp, round_cols, engin_date, multivariable_lag
from .tsf_model_selection import model_prediction
from .tsf_model_configs import model_configurations

h_parameters=model_configurations()

def pred_dataset(Dataset:pd.DataFrame,
                 forecast_size:int,
                 granularity:str='1d'):
    """
    The pred_dataset function takes a dataset and returns a dataframe with the timestamp column 
    extended to include the next forecast_size number of rows. The granularity parameter determines 
    how far apart each row is timewise. For example, if granularity='30m' then each new row will be 30 minutes 
    apart from the previous row.
    
    :param Dataset:pd.DataFrame: Pass the dataset that is to be used for forecasting
    :param forecast_size:int: Specify the number of rows in the forecast dataset
    :param granularity:str='1d': Specify the frequency of the time series
    :return: A dataframe with the timestamp of the last row of our dataset, and a number of rows equal to forecast_size
    """
    Dataframe=Dataset.copy()
    Dataframe=slice_timestamp(Dataframe)

    timestamp,timestamp_list=list((Dataframe.Date[len(Dataframe)-1:]))[0],[]
    
    if granularity=='1m':
    
        def generate_datetimes(date_from_str=timestamp, days=1000):
           date_from = datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
           for n in range(1,1420*days):
               yield date_from + datetime.timedelta(minutes=n)
        for date in generate_datetimes():
            timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))
            
    elif granularity=='30m':
    
        def generate_datetimes(date_from_str=timestamp, days=1000):
           date_from = datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
           for n in range(1,1420*days):
               yield date_from + datetime.timedelta(minutes=30*n)
        for date in generate_datetimes():
            timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))
    
    elif granularity=='1h':
    
        def generate_datetimes(date_from_str=timestamp, days=1000):
           date_from = datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
           for hour in range(1,24*days):
               yield date_from + datetime.timedelta(hours=hour)
        for date in generate_datetimes():
            timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))
        
    elif granularity=='1d':
    
        def generate_datetimes(date_from_str=timestamp, days=1000):
           date_from = datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
           for day in range(1,days):
               yield date_from + datetime.timedelta(days=day)
        for date in generate_datetimes():
            timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))

    elif granularity=='1wk':
            
        def generate_datetimes(date_from_str=timestamp, weeks=1000):
           date_from = datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
           for week in range(1,weeks):
               yield date_from + datetime.timedelta(weeks=week)
        for date in generate_datetimes():
            timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))
    
    elif granularity=='1mo':
    
        def generate_datetimes(date_from_str=timestamp, months=100):
           date_from = datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
           for month in range(0,months):
               yield date_from + relativedelta(months=month+1)
        for date in generate_datetimes():
            timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))
    
    df = pd.DataFrame()
    df['Date']=timestamp_list
    df['Date']=pd.to_datetime(df['Date'])
    
    df=df.iloc[0:forecast_size,:]
    
    return df

def pred_results(Dataset:pd.DataFrame,
                 forecast_size:int,
                 model_configs:dict=h_parameters,
                 granularity:str='1d',
                 selected_model:str='RandomForest'):
    """
    The pred_results function takes in a dataset, the name of the model to be used for prediction, 
    the forecast size and a dictionary containing all parameters needed to run the selected model.
    The function returns a dataframe with real values and predicted values from the specified model.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be used for prediction
    :param selected_model:str: Select the model that will be used for the prediction
    :param forecast_size:int: Define the number of periods to forecast
    :param model_configs:dict: Pass the parameters of the model
    :param granularity:str: Define the time interval of the data
    :return: A dataframe with the predicted values
    """
    Dataset=slice_timestamp(Dataset)
    df_pred=pred_dataset(Dataset,forecast_size,granularity)
    Dataset["Values"]=0
    df_pred["Values"]=1
    df_final= pd.concat([Dataset,df_pred])
    df_final.index=df_final['Date']
    
    list_mv=['RandomForest','ExtraTrees','GBR','KNN','GeneralizedLR','AutoKeras','XGBoost','H2O_AutoML']
    list_uv=['Prophet','AutoArima']
    
    target='y'

    if selected_model in list_uv:
        
        df=Dataset.copy()
        df = df.rename(columns={'Date': 'ds'})
        df = df[['ds', target]]
        
        if granularity=="1m":
            frequency='min'
        elif granularity=="30m":
            frequency='30min'
        elif granularity=="1h":
            frequency='H'
        elif granularity=="1d":
            frequency='D'
        elif granularity=="1wk":
            frequency='W'
        elif granularity=="1mo":
            frequency='M'
            
        if selected_model =='Prophet':
            pr_params=model_configs['Prophet']
            m = Prophet(**pr_params)
            model_p = m.fit(df)
            
            future = model_p.make_future_dataframe(periods=forecast_size,freq=frequency)
            forecast = model_p.predict(future)
            col='yhat'
            y_pred=forecast.iloc[len(forecast)-forecast_size:,:]
            
        elif selected_model =='AutoArima':
            aa_params=model_configs['AutoArima']
            model_arima=auto_arima(df[['y']], **aa_params)
            y_pred = model_arima.predict(n_periods=forecast_size)
            y_pred=pd.DataFrame(y_pred, columns = [0])
            col=0
        
        y_pred=y_pred[col]
        
        df_final['y'][len(df_final)-forecast_size:]=y_pred
        df_final['Values']=df_final['Values'].replace(0,'Real')
        df_final['Values']=df_final['Values'].replace(1,'Predicted')
        df_final=round_cols(df_final, target)

    elif selected_model in list_mv:

        df_final=engin_date(df_final,Drop=False)
        df_final=multivariable_lag(df_final,target,range_lags=[forecast_size,forecast_size*3]) 
        
        Train=df_final.iloc[:len(df_final)-forecast_size,:]
        Test=df_final.iloc[len(df_final)-forecast_size:,:]

        input_cols=list(df_final.columns)
        input_cols.remove(target)  
        
        scaler = MinMaxScaler()
        scaler = scaler.fit(Train[input_cols])
        
        Train[input_cols] = scaler.transform(Train[input_cols])
        Test[input_cols] = scaler.transform(Test[input_cols])

        y_pred=model_prediction(Train, Test,target,model_configs=model_configs,algo=selected_model)
        df_final['y'][len(df_final)-forecast_size:]=y_pred
        df_final['Values']=df_final['Values'].replace(0,'Real')
        df_final['Values']=df_final['Values'].replace(1,'Predicted')
        df_final['Date']=df_final.index
        df_final=df_final[['Date','y','Values']]
        df_=Dataset.head(forecast_size*3)
        df_.index=df_.Date
        df_['Values']='Real'

        df_final = pd.concat([df_,df_final])
        df_final = df_final.reset_index(drop=True)
        
    return df_final
