import pandas as pd
import numpy as np
import sys
import datetime
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from pmdarima.arima import auto_arima
from .tsf_model_selection import model_prediction
from .tsf_data_eng import slice_timestamp, engin_date, multivariable_lag
from .tsf_model_configs import model_configurations

h_parameters=model_configurations()

def Univariate_Forecast(dataset:pd.DataFrame,
                        train_length:float,
                        forecast_length:int,
                        rolling_window_size:int,
                        algo:str,
                        model_configs:dict=h_parameters,
                        granularity:str="1d"):
    """
    The Univariate_Forecast function takes a dataset and makes a forecast for the target variable.
    The function returns a dataframe with the predictions of the target variable.
    Parameters: dataset, train_length, forecast_length, rolling_window_size, algo (Prophet or AutoArima), model configs (dict) and granularity(int). 
    
    :param dataset:pd.DataFrame: Pass the dataset to be used for forecasting
    :param train_length:float: Specify the percentage of data to be used as train
    :param forecast_length:int: Indicate the number of periods to forecast
    :param rolling_window_size:int: Define the size of the window used to evaluate the performance of each algorithm
    :param algo:str: Select the forecasting algorithm to be used
    :param model_configs:dict: Configure the models
    :param granularity:str=&quot;1m&quot;: Set the granularity of the time series
    :return: A dataframe with the predictions
    """
    dfs_list,list_y_true,list_y_pred=[],[],[]
    
    target='y'
    
    dataset=slice_timestamp(dataset)
    dataset_=dataset.copy()
    dataset_=dataset_.iloc[forecast_length*3:,:]  ## Equal data lenght to Multivariate Approach given the rows drop in multivariate lags*  
    
    df_final = dataset_[['Date', target]]
    df_final = df_final.rename(columns={'Date': 'ds'}) 
    
    size_train=int((train_length*len(df_final)))
    
    train=df_final.iloc[:size_train,:]
    test=df_final.iloc[size_train:,:]
    
    train[target]=train[target].astype(np.int64)
    test[target]=test[target].astype(np.int64)
    
    assert len(test)>=forecast_length , "forecast_length>=len(Test), try to reduce your train_size ratio"
    
    iterations,iters = (int((len(test))/rolling_window_size)),(int((len(test)-forecast_length)/rolling_window_size))+1
    
    if algo=='AutoArima':
    
        aa_params=model_configs['AutoArima']
        model_arima=auto_arima(train[[target]], **aa_params)
    
    for rolling_cycle in range(0, iterations):
        
        if rolling_cycle==0:
            window=0
        else:
            window=rolling_window_size
        
        size_train=size_train+window
        
        train=df_final.iloc[:size_train,:]
        test=df_final.iloc[size_train:,:]
        
        if len(test[target])>=forecast_length:
            
            print('Algorithm Evaluation:', algo,'|| Window Iteration:', rolling_cycle + 1, "of", iters)
            
            print('Rows Train:', len(train))
            
            print('Rows Test:', len(test))

            train[[target]]=train[[target]].astype('int32')
            
            train_=train.copy()
            train_=train_[['ds', target]]
            
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
            
            if algo=='Prophet':
                pr_params=model_configs['Prophet']
                m = Prophet(**pr_params)
                model_p = m.fit(train_)
                
                future = model_p.make_future_dataframe(periods=forecast_length,freq=frequency)
                forecast = model_p.predict(future)

                col='yhat'
                y_pred=forecast.iloc[len(forecast)-forecast_length:,:]

            elif algo=='AutoArima':
                
                model_arima.fit(train_[[target]])
            
                forecast = model_arima.predict(n_periods=forecast_length)
                y_pred=pd.DataFrame(forecast, columns = [0]) 
                col=0
                
            y_pred=y_pred[col]
                        
            y_true=test[target][0:forecast_length]
            
            y_pred_list = y_pred.tolist()
            y_true_list = y_true.tolist()

            list_y_true.append(y_true)
            list_y_pred.append(y_pred)
            x,y=pd.concat(list_y_true),pd.concat(list_y_pred) 
            x,y=x.reset_index(drop=True),y.reset_index(drop=True)
            df = pd.DataFrame()
            df = pd.DataFrame({'y_true': x,'y_pred': y,'Window':rolling_cycle})
            df_=df.iloc[len(df)-forecast_length:,:]
            
            df_.index=test['ds'][0:forecast_length]
            dfs_list.append(df_)

    df_pred=pd.concat(dfs_list)
    df_pred['Window']=df_pred['Window']+1

    return df_pred


def Multivariate_Forecast(dataset:pd.DataFrame,
                          train_length:float,
                          forecast_length:int,
                          rolling_window_size:int,
                          algo:str,
                          model_configs:dict=h_parameters,
                          granularity:str='1d'):
    """
    The Multivariate_Forecast function takes a dataset, the length of the train period, 
    the forecast length and a rolling window size as input. It then splits the data into train and test sets. 
    It then applies an algorithm to predict future values based on past patterns in the data. The function returns 
    a DataFrame with actual values (y_true) and predicted values (y_pred). 
    
    :param dataset:pd.DataFrame: Pass the dataset to be used in the model
    :param train_length:float: Define the length of the train dataset
    :param forecast_length:int: Define the number of periods to forecast
    :param rolling_window_size:int: Define the size of the window that will be used to split the dataset
    :param algo:str: Select the algorithm to be used for training and forecasting
    :param model_configs:dict: Configure the model
    :param granularity:str='1m': Define the time interval of the data
    :return: A dataframe with the predicted values and the window number
    """
    assert train_length>=0.3 and train_length<=1 , "train_length value should be in [0.3,1[ interval"

    dfs_list,list_y_true,list_y_pred=[],[],[]
    
    target='y'   
    
    dataset_=dataset.copy()
    dataset_=slice_timestamp(dataset_)
    
    dataset_=engin_date(dataset_)

    dataset_ = multivariable_lag(dataset_,target,range_lags=[forecast_length,forecast_length*3])

    df_final = dataset_.copy()
    
    size_train=int((train_length*len(df_final)))

    train=df_final.iloc[:size_train,:]
    test=df_final.iloc[size_train:,:]
    
    train[target]=train[target].astype(np.int64)
    test[target]=test[target].astype(np.int64)    
    
    assert len(test)>=forecast_length , "forecast_length>=len(Test), try to reduce your train_size ratio"
    
    iterations,iters = (int((len(test))/rolling_window_size)),(int((len(test)-forecast_length)/rolling_window_size))+1
      
    for rolling_cycle in range(0, iterations):
            
        if rolling_cycle==0:
            window=0
        else:
            window=rolling_window_size
        
        size_train=size_train+window
        
        train=df_final.iloc[:size_train,:]
        test=df_final.iloc[size_train:,:]
        
        if (len(test[target])>=forecast_length):
            
            print('Algorithm Evaluation:', algo,'|| Window Iteration:', rolling_cycle + 1, "of", iters)
            
            print('Rows Train:', len(train))
            
            print('Rows Test:', len(test))
            
            train[[target]]=train[[target]].astype('int32')
            
            input_cols=list(train.columns)
            input_cols.remove(target)

            scaler = MinMaxScaler() 

            scaler = scaler.fit(train[input_cols])
            train[input_cols] = scaler.transform(train[input_cols])
            test[input_cols] = scaler.transform(test[input_cols])            

            y_pred=model_prediction(train, test,target,model_configs=model_configs,algo=algo)
            y_pred=y_pred[0:forecast_length]
            y_pred= pd.Series(y_pred.tolist())
            y_pred_list = y_pred.tolist()
            
            y_true=test[target][0:forecast_length]
            y_true = pd.Series(y_true)
            y_true_list = y_true.tolist()
            
            y_t_axys=list(test.index[0:forecast_length])
            
            list_y_true.append(y_true)
            list_y_pred.append(y_pred)
            x,y=pd.concat(list_y_true),pd.concat(list_y_pred) 
            x,y=x.reset_index(drop=True),y.reset_index(drop=True)
            df = pd.DataFrame()
            df = pd.DataFrame({'y_true': x,'y_pred': y,'Window':rolling_cycle})
            df_=df.iloc[len(df)-forecast_length:,:]
            df_.index=test.index[0:forecast_length]

            dfs_list.append(df_)

    df_pred=pd.concat(dfs_list)

    df_pred['Window'] = df_pred['Window']+1
    df_pred['y_pred'] = df_pred['y_pred'].astype(str) 
    df_pred['y_pred'] = df_pred['y_pred'].apply(lambda x: x.replace('[', ''))
    df_pred['y_pred'] = df_pred['y_pred'].apply(lambda x: x.replace(']', ''))
    df_pred['y_pred'] = pd.to_numeric(df_pred['y_pred'])
    df_pred['y_pred'] = df_pred['y_pred'].round(4)

    return df_pred

