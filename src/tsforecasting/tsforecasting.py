import pandas as pd
import numpy as np
import sys
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import TweedieRegressor
from fbprophet import Prophet
from neuralprophet import NeuralProphet
from pmdarima.arima import auto_arima
import autokeras as ak
import xgboost
import h2o
from h2o.automl import H2OAutoML
#import tensorflow as tf

Model_Configs ={'RandomForest':{'n_estimators':250,'random_state':42,'criterion':"squared_error",
                   'max_depth':None,'max_features':"auto"},
               'ExtraTrees':{'n_estimators':250,'random_state':42,'criterion':"squared_error",
                   'max_depth':None,'max_features':"auto"}, 
               'GBR':{'n_estimators':250,'learning_rate':0.1,'criterion':"friedman_mse",
                    'max_depth':3,'min_samples_split':5,'learning_rate':0.01,'loss':'ls'},
               'KNN':{'n_neighbors': 3,'weights':"uniform",'algorithm':"auto",'metric_params':None},
               'GeneralizedLR':{'power':1,'alpha':0.5,'link':'log','fit_intercept':True,
                    'max_iter':100,'warm_start':False,'verbose':0},
               'XGBoost':{'objective':'reg:squarederror','n_estimators':1000,'nthread':24},
               'H2O_AutoML':{'max_models':50,'nfolds':0,'seed':1,'max_runtime_secs':30,
                    'sort_metric':'AUTO','exclude_algos':['GBM','DeepLearning']},
               'AutoKeras':{'max_trials':1,'overwrite':42,'loss':"mean_squared_error",
                    'max_model_size':None,'epochs':50},
               'AutoArima':{'start_p':0, 'd':1, 'start_q':0,'max_p':3, 'max_d':3, 'max_q':3,
                    'start_P':0, 'D':1, 'start_Q':0, 'max_P':3, 'max_D':3,'max_Q':3,
                    'm':1,'trace':True, 'seasonal':True,'random_state':20,'n_fits':10},
               'Prophet':{'growth':'linear','changepoint_range':0.8,'yearly_seasonality':'auto',
                    'weekly_seasonality':'auto','daily_seasonality':'auto',},
               'NeuralProphet':{'growth':'linear','n_lags':0,'yearly_seasonality':'auto',
                    'weekly_seasonality':'auto','daily_seasonality':'auto','n_forecasts':1,
                    'epochs':None,'num_hidden_layers':0,'loss_func':"Huber",'optimizer':"AdamW"}
                }

def  reset_index_DF(Dataset:pd.DataFrame):
    
    Dataset=Dataset.reset_index()
    Dataset.drop(Dataset.columns[0], axis=1, inplace=True)
    
    return Dataset

def slice_timestamp(Dataset:pd.DataFrame,date_col:str='Date'):
    """
    The slice_timestamp function takes a dataframe and returns the same dataframe with the date column sliced to just include
    the year, month, day and hour. This is done by converting all of the values in that column to strings then slicing them 
    accordingly. The function then converts those slices back into datetime objects so they can be used for further analysis.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be sliced
    :param date_col:str='Date': Specify the name of the column that contains the date information
    :return: A dataframe with the timestamp column sliced to only include the year, month and day
    """
    Dataframe=Dataset.copy()
    cols=list(Dataframe.columns)
    for col in cols:
        if col==date_col:
            Dataframe[date_col] = Dataframe[date_col].astype(str)
            Dataframe[date_col] = Dataframe[date_col].str.slice(0,19)
            Dataframe[date_col] = pd.to_datetime(Dataframe[date_col])
    return Dataframe

def round_cols(Dataset:pd.DataFrame,
               target,round_:int=4):
    """
    The round_cols function rounds the numeric columns of a Dataset to a specified number of decimal places.
    The function takes two arguments:
    Dataset: A pandas DataFrame object.
    target: The name of the target column in the dataframe that will not be rounded.  This argument is required for safety reasons, so that no other columns are accidentally rounded.
    
    :param Dataset:pd.DataFrame: Pass the dataframe that will be transformed
    :param target: Indicate the target variable
    :param round_:int=4: Round the numbers to a certain number of decimal places
    :return: A dataframe with the same columns as the original one, but with all numeric columns rounded to 4 decimals
    """
    Dataframe_=Dataset.copy()
    Df_round=Dataframe_.copy()
    list_num_cols=Df_round.select_dtypes(include=['float']).columns.tolist()
    
    for elemento in list_num_cols:
        if elemento==target:
            list_num_cols.remove(target)
    for col in list_num_cols:
        Df_round[[col]]=Df_round[[col]].round(round_)
        
    return Df_round

def engin_date(Dataset:pd.DataFrame,
               Drop:bool=False):

    """
    The engin_date function takes a DataFrame and returns a DataFrame with the date features engineered.
    The function has two parameters: 
    Dataset: A Pandas DataFrame containing at least one column of datetime data. 
    Drop: A Boolean value indicating whether or not to drop the original datetime columns from the returned dataset.
    
    :param Dataset:pd.DataFrame: Pass the dataset
    :param Drop:bool=False: Drop the original timestamp columns
    :return: The dataframe with the date features generated
    """
    Dataset_=Dataset.copy()
    Df=Dataset_.copy()
    Df=slice_timestamp(Df)
    
    x=pd.DataFrame(Df.dtypes)
    x['column'] = x.index
    x=x.reset_index().drop(['index'], axis=1).rename(columns={0: 'dtype'})
    a=x.loc[x['dtype'] == 'datetime64[ns]']

    list_date_columns=[]
    for col in a['column']:
        list_date_columns.append(col)

    def create_date_features(df,elemento):
        
        df[elemento + '_day_of_month'] = df[elemento].dt.day
        df[elemento + '_day_of_week'] = df[elemento].dt.dayofweek + 1
        df[[elemento + '_is_wknd']] = df[[elemento + '_day_of_week']].replace([1, 2, 3, 4, 5, 6, 7], 
                            [0, 0, 0, 0, 0, 1, 1 ]) 
        df[elemento + '_month'] = df[elemento].dt.month
        df[elemento + '_day_of_year'] = df[elemento].dt.dayofyear
        df[elemento + '_year'] = df[elemento].dt.year
        df[elemento + '_hour']=df[elemento].dt.hour
        df[elemento + '_minute']=df[elemento].dt.minute
        df[elemento + '_Season']=''
        winter = list(range(1,80)) + list(range(355,370))
        spring = range(80, 172)
        summer = range(172, 264)
        fall = range(264, 355)

        df.loc[(df[elemento + '_day_of_year'].isin(spring)), elemento + '_Season'] = '2'
        df.loc[(df[elemento + '_day_of_year'].isin(summer)), elemento + '_Season'] = '3'
        df.loc[(df[elemento + '_day_of_year'].isin(fall)), elemento + '_Season'] = '4'
        df.loc[(df[elemento + '_day_of_year'].isin(winter)), elemento + '_Season'] = '1'
        df[elemento + '_Season']=df[elemento + '_Season'].astype(np.int64)
        
        return df 
    
    if Drop==True:
        for elemento in list_date_columns:
            Df=create_date_features(Df,elemento)
            Df=Df.drop(elemento,axis=1)
    elif Drop==False:
        for elemento in list_date_columns:
            Df=create_date_features(Df,elemento)
    #if len(list_date_columns)>=1:
    #    print('Date Time Feature Generation')
        
    return Df

def multivariable_lag(Dataset:pd.DataFrame,
                      target:str="y",
                      range_lags:list=[1,10],
                      drop_na:bool=True):
    """
    The multivariable_lag function takes a Pandas DataFrame and returns a new DataFrame with the target variable 
    lagged by the number of periods specified in range_lags. The function also removes NaN values from the dataset, 
    and can be used to remove all NaN values or just those at the end of a time series. The default is to drop rows with any 
    NaNs, but this can be changed by setting drop_na=False.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be used in the function
    :param target:str: Indicate the name of the column that will be used as target
    :param range_lags:list=[1: Define the range of lags to be used
    :param 10]: Indicate the maximum lag to be used
    :param drop_na:bool=True: Drop the rows with nan values
    :return: A dataframe with the lags specified by the user
    """
    assert range_lags[0]>=1, "Range lags first interval value should be bigger then 1"
    assert range_lags[0]<=range_lags[1], "Range lags first interval value should be bigger then second"
    
    Df=Dataset.copy()
    Dataframe=Df.copy()
    Dataframe=slice_timestamp(Dataframe)
    
    lag_str,list_dfs='',[]

    for elemento in range(range_lags[0],range_lags[1]+1):
        lag_str=str(elemento)
        lag_str='lag_'+lag_str
        Df_Final=pd.DataFrame({'target': Df[target],
                      lag_str: Df[target].shift(elemento)
                      })
        Df_Final=Df_Final.drop('target',axis=1)
        Dataframe[lag_str]=Df_Final[lag_str]
    cols,Cols_Input=list(Dataframe.columns),list(Dataframe.columns)
    Cols_Input.remove(target)
    last_col=cols[len(cols)-1:]
    if drop_na==True:
        Dataframe = Dataframe.dropna(axis=0, subset=last_col)
    elif drop_na==False:
        Dataframe[Cols_Input]=Dataframe[Cols_Input].apply(lambda x: x.fillna(x.mean()),axis=0)
    for col in cols:
        if col=='Date':
            Dataframe=Dataframe.set_index(['Date']) 
            break
    return Dataframe

def feature_selection_tb(Dataset:pd.DataFrame,
                         target:str="y",
                         total_vi:float=0.99,
                         algo:str="ExtraTrees",
                         estimators:int=250):
    """
    The feature_selection_tb function takes in a pandas dataframe and returns the selected columns.
    The function uses ExtraTreesRegressor to select the most important features from a dataset. 
    The user can specify how much of the total variance they want explained by their model, as well as what algorithm they would like to use for feature selection.
    
    :param Dataset:pd.DataFrame: Pass the dataset
    :param target:str=&quot;y&quot;: Specify the target variable name
    :param total_vi:float=0.99: Set the total variable importance that needs to be achieved
    :param algo:str=&quot;ExtraTrees&quot;: Select the algorithm to be used for feature selection
    :param estimators:int=250: Set the number of estimators in the randomforest and extratrees algorithms
    :return: A list of columns that are selected by the algorithm
    """
    assert total_vi>=0.5 and total_vi<=1 , "total_vi value should be in [0.5,1[ interval"
    
    Train=Dataset.copy()
    Selected_Cols= list(Train.columns)
    Selected_Cols.remove(target)
    Selected_Cols.append(target)
    Train=Train[Selected_Cols]
   
    X_train = Train.iloc[:, 0:(len(Selected_Cols)-1)].values
    y_train = Train.iloc[:, (len(Selected_Cols)-1)].values
    
    if algo=='ExtraTrees':
        fs_model = ExtraTreesRegressor(n_estimators=estimators)
        fs_model.fit(X_train, y_train)
    elif algo=="RandomForest":
        fs_model = RandomForestRegressor(n_estimators=estimators)
        fs_model.fit(X_train, y_train)
    elif algo=='GBR':
        fs_model = GradientBoostingRegressor(n_estimators=estimators)
        fs_model.fit(X_train, y_train)

    column_imp=fs_model.feature_importances_
    column_names=Selected_Cols.copy()
    column_names.remove(target)
    
    Feat_imp = pd.Series(column_imp)
    Columns = pd.Series(column_names)
    
    df=pd.concat([Feat_imp,Columns],axis=1)
    df = df.rename(columns={0: 'percentage',1: 'variable'})
    
    n=0.015
    Va_df = df[df['percentage'] > n]
    Variable_Importance=Va_df['percentage'].sum()
    for iteration in range(0,10):
            
        if Variable_Importance<=total_vi:
            Va_df = df[df['percentage'] > n]
            n=n*0.5
            Variable_Importance=Va_df['percentage'].sum()
        elif Variable_Importance>total_vi:
            break
    Va_df = Va_df.sort_values(["percentage"], ascending=False)
    #print("Approximate minimum value of Relative Percentage:",n)
    Selected_Columns=[]
    for rows in Va_df["variable"]:
        Selected_Columns.append(rows)
    Selected_Columns.append(target)
    
    return Selected_Columns, Va_df

def metrics_regression(y_real, y_prev):
    """
    The metrics_regression function calculates the metrics for regression models.
    It takes as input two arrays: y_real and y_prev, which are the real values 
    of a target variable and its predictions respectively. The function returns 
    a dictionary with all the metrics.
    
    :param y_real: Store the real values of the target variable
    :param y_prev: Compare the real values of y with the predicted values
    :return: A dictionary with the metrics of the regression model
    """
    mae=mean_absolute_error(y_real, y_prev)
    mape = (mean_absolute_percentage_error(y_real, y_prev))*100
    mse=mean_squared_error(y_real, y_prev)
    evs= explained_variance_score(y_real, y_prev)
    maximo_error= max_error(y_real, y_prev)
    metrics_pred_regression= {'Mean Absolute Error': mae, 
                              'Mean Absolute Percentage Error': mape,
                              'Mean Squared Error': mse,
                              'Explained Variance Score': evs, 
                              'Max Error': maximo_error}
    
    return metrics_pred_regression

def model_prediction(Train:pd.DataFrame,
                     Test:pd.DataFrame,
                     target:str="y",
                     model_configs:dict=Model_Configs,
                     algo:str='RandomForest'):
    """
    The model_prediction function takes in a Train and Test dataframe, as well as the name of the target column. 
    It then trains a model using the specified algorithm (RandomForest by default) and returns predictions for 
    the test set.
    
    :param Train:pd.DataFrame: Pass the train dataframe
    :param Test:pd.DataFrame: Test the model with a different dataset
    :param target:str: Specify the column name of the target variable
    :param model_configs:dict: Pass the parameters of each model
    :param algo:str='RandomForest': Select the model to be used
    :return: The predictions of the model
    """
    Selected_Cols= list(Train.columns)
    Selected_Cols.remove(target)
    Selected_Cols.append(target) 
    Train=Train[Selected_Cols]
    Test=Test[Selected_Cols]   
    
    X_train = Train.iloc[:, 0:(len(Selected_Cols)-1)].values
    X_test = Test.iloc[:, 0:(len(Selected_Cols)-1)].values
    y_train = Train.iloc[:, (len(Selected_Cols)-1)].values
    y_test = Test.iloc[:, (len(Selected_Cols)-1)].values
    
    if algo=='RandomForest':
        rf_params=model_configs['RandomForest']
        regressor_RF = RandomForestRegressor(**rf_params) 
        regressor_RF.fit(X_train, y_train)
        y_predict = regressor_RF.predict(X_test)
        
    elif algo=='ExtraTrees':
        et_params=model_configs['ExtraTrees']
        regressor_ET = ExtraTreesRegressor(**et_params)
        regressor_ET.fit(X_train, y_train)
        y_predict = regressor_ET.predict(X_test)
        
    elif algo=='GBR':
        gbr_params=model_configs['GBR']
        regressor_GBR = GradientBoostingRegressor(**gbr_params)
        regressor_GBR.fit(X_train, y_train)
        y_predict = regressor_GBR.predict(X_test)
        
    elif algo=='KNN':
        knn_params=model_configs['KNN']
        regressor_KNN = KNeighborsRegressor(**knn_params)
        regressor_KNN.fit(X_train, y_train)
        y_predict = regressor_KNN.predict(X_test) 
        
    elif algo=='GeneralizedLR':
        td_params=model_configs['GeneralizedLR']
        regressor_TD = TweedieRegressor(**td_params)
        regressor_TD.fit(X_train, y_train)
        y_predict = regressor_TD.predict(X_test)
        
    elif algo=='H2O_AutoML':
        Test[target]=Test[target].fillna(0) ## Avoid H2O OS_Error
        Test[target]=Test[target].astype(np.float) ## Avoid H2O OS_Error
        Train_h2o,Test_h2o=h2o.H2OFrame(Train),h2o.H2OFrame(Test)
        Input_Cols=Selected_Cols.copy()
        Input_Cols.remove("y")
        aml_params=model_configs['H2O_AutoML']
        aml = H2OAutoML(**aml_params) 
        aml.train(x=Input_Cols ,y=target ,training_frame=Train_h2o)
        
        leaderboards = aml.leaderboard
        leaderboard_df= leaderboards.as_data_frame()
        Id_Modelo_Lider=leaderboard_df['model_id'][0]
          
        H2O_Modelo_Lider=h2o.get_model(Id_Modelo_Lider)
        
        coluna_previsao = H2O_Modelo_Lider.predict(Test_h2o)
        coluna_previsao = coluna_previsao.asnumeric()
        y_predict= coluna_previsao.as_data_frame()['predict']
    elif algo=='XGBoost':
        xg_params=model_configs['XGBoost']
        regressor_XG = xgboost.XGBRegressor(**xg_params)
        regressor_XG.fit(X_train, y_train)
        y_predict = regressor_XG.predict(X_test)
        
    elif algo=='AutoKeras':
        epochs=model_configs['AutoKeras']['epochs']
        ak_params=model_configs['AutoKeras']
        del ak_params['epochs']
        regressor_AK = ak.StructuredDataRegressor(**ak_params)
        regressor_AK.fit(X_train, y_train, epochs=epochs)
        y_predict = regressor_AK.predict(X_test)
        model_configs['AutoKeras']['epochs']=epochs
   
    return y_predict

def Multivariate_Forecast(Dataset:pd.DataFrame,
                          train_length:float,
                          forecast_length:int,
                          rolling_window_size:int,
                          algo:str,
                          model_configs:dict,
                          granularity:str='1d'):
    """
    The Multivariate_Forecast function takes a dataset, the length of the train period, 
    the forecast length and a rolling window size as input. It then splits the data into train and test sets. 
    It then applies an algorithm to predict future values based on past patterns in the data. The function returns 
    a DataFrame with actual values (y_true) and predicted values (y_pred). 
    
    :param Dataset:pd.DataFrame: Pass the dataset to be used in the model
    :param train_length:float: Define the length of the train dataset
    :param forecast_length:int: Define the number of periods to forecast
    :param rolling_window_size:int: Define the size of the window that will be used to split the dataset
    :param algo:str: Select the algorithm to be used for training and forecasting
    :param model_configs:dict: Configure the model
    :param granularity:str='1m': Define the time interval of the data
    :return: A dataframe with the predicted values and the window number
    """
    assert train_length>=0.3 and train_length<=1 , "train_length value should be in [0.3,1[ interval"

    list_completa_dfs,list_y_true,list_y_pred=[],[],[]
    
    target='y'   
    
    Df=Dataset.copy()
    Df=slice_timestamp(Df)
    
    Df=engin_date(Df)

    Df = multivariable_lag(Df,target,range_lags=[forecast_length,forecast_length*3])

    Df_Final = Df.copy()
    
    size_Train=int((train_length*len(Df_Final)))

    Train=Df_Final.iloc[:size_Train,:]
    Test=Df_Final.iloc[size_Train:,:]
    
    Train[target]=Train[target].astype(np.int64)
    Test[target]=Test[target].astype(np.int64)    
    
    assert len(Test)>=forecast_length , "forecast_length>=len(Test), try to reduce your train_size ratio"
    
    iterations = (int((len(Test))/rolling_window_size))
      
    for rolling_cycle in range(0, iterations):
            
        if rolling_cycle==0:
            window=0
        else:
            window=rolling_window_size
        
        size_Train=size_Train+window
        
        Train=Df_Final.iloc[:size_Train,:]
        Test=Df_Final.iloc[size_Train:,:]
        
        if (len(Test[target])>=forecast_length):
            
            print('Algorithm Evaluation:', algo,'|| Window Iteration:', rolling_cycle + 1, "of", iterations)
            
            print('Rows Train:', len(Train))
            
            print('Rows Test:', len(Test))
            
            Train[[target]]=Train[[target]].astype('int32')
            
            input_cols=list(Train.columns)
            input_cols.remove(target)

            scaler = MinMaxScaler() ## -> Encoding Application

            scaler = scaler.fit(Train[input_cols])
            Train[input_cols] = scaler.transform(Train[input_cols])
            Test[input_cols] = scaler.transform(Test[input_cols])            

            y_pred=model_prediction(Train, Test,target,model_configs,algo)
            y_pred=y_pred[0:forecast_length]
            y_pred= pd.Series(y_pred.tolist())
            y_pred_list = y_pred.tolist()
            
            y_true=Test[target][0:forecast_length]
            y_true = pd.Series(y_true)
            y_true_list = y_true.tolist()
            
            y_t_axys=list(Test.index[0:forecast_length])
            
            """
            if barplots==True:
                plt.plot(y_true_list, label='Actual')
                plt.plot(y_pred_list, label='Predicted')
                plt.plot(ylabel=y_t_axys)
                plt.title(f"{algo} || Window Iteration: {rolling_cycle+1} of {iterations}.", loc='center')
                plt.legend()
                plt.show()
            """
            list_y_true.append(y_true)
            list_y_pred.append(y_pred)
            x=pd.concat(list_y_true)
            y=pd.concat(list_y_pred)
            x=reset_index_DF(x)
            y=reset_index_DF(y)
            df = pd.DataFrame()
            df[['y_true']]=x
            df[['y_pred']]=y
            
            df[['Window']]=rolling_cycle
            df_=df.iloc[len(df)-forecast_length:,:]
            df_.index=Test.index[0:forecast_length]

            list_completa_dfs.append(df_)

    Df_Pred=pd.concat(list_completa_dfs)

    Df_Pred['Window']=Df_Pred['Window']+1
    Df_Pred['y_pred'] = Df_Pred['y_pred'].astype(str) 
    Df_Pred['y_pred'] = Df_Pred['y_pred'].apply(lambda x: x.replace('[', ''))
    Df_Pred['y_pred'] = Df_Pred['y_pred'].apply(lambda x: x.replace(']', ''))
    Df_Pred['y_pred'] = pd.to_numeric(Df_Pred['y_pred'])
    Df_Pred['y_pred'] = Df_Pred['y_pred'].round(3)

    return Df_Pred

def Univariate_Forecast(Dataset:pd.DataFrame,
                        train_length:float,
                        forecast_length:int,
                        rolling_window_size:int,
                        algo:str,
                        model_configs:dict,
                        granularity:str="1d"):
    """
    The Univariate_Forecast function takes a dataset and makes a forecast for the target variable.
    The function returns a dataframe with the predictions of the target variable.
    Parameters: Dataset, train_length, forecast_length, rolling_window_size, algo (Prophet or AutoArima), model configs (dict) and granularity(int). 
    
    :param Dataset:pd.DataFrame: Pass the dataset to be used for forecasting
    :param train_length:float: Specify the percentage of data to be used as train
    :param forecast_length:int: Indicate the number of periods to forecast
    :param rolling_window_size:int: Define the size of the window used to evaluate the performance of each algorithm
    :param algo:str: Select the forecasting algorithm to be used
    :param model_configs:dict: Configure the models
    :param granularity:str=&quot;1m&quot;: Set the granularity of the time series
    :return: A dataframe with the predictions
    """
    list_completa_dfs,list_y_true,list_y_pred=[],[],[]
    
    target='y'
    
    Dataset=slice_timestamp(Dataset)
    train_split_df=Dataset.copy()
    Dataframe=Dataset.copy()
        
    Df_Final = Dataframe[['Date', target]]
    Df_Final = Df_Final.rename(columns={'Date': 'ds'}) 
    
    size_Train=int((train_length*len(Df_Final)))
    
    Train=Df_Final.iloc[:size_Train,:]
    Test=Df_Final.iloc[size_Train:,:]
    
    Train[target]=Train[target].astype(np.int64)
    Test[target]=Test[target].astype(np.int64)
    
    assert len(Test)>=forecast_length , "forecast_length>=len(Test), try to reduce your train_size ratio"
    
    iterations = (int((len(Test))/rolling_window_size))
    
    if algo=='AutoArima':
    
        aa_params=model_configs['AutoArima']
        model_arima=auto_arima(Train[[target]], **aa_params)
    
    for rolling_cycle in range(0, iterations):
        
        if rolling_cycle==0:
            window=0
        else:
            window=rolling_window_size
        
        size_Train=size_Train+window
        
        Train=Df_Final.iloc[:size_Train,:]
        Test=Df_Final.iloc[size_Train:,:]
        
        if len(Test[target])>=forecast_length:
            
            print('Algorithm Evaluation:', algo,'|| Window Iteration:', rolling_cycle + 1, "of", iterations)
            
            print('Rows Train:', len(Train))
            
            print('Rows Test:', len(Test))

            Train[[target]]=Train[[target]].astype('int32')
            
            train_=Train.copy()
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
                
            elif algo=='NeuralProphet':
                np_params=model_configs['NeuralProphet']
                model_np = NeuralProphet(**np_params)
                freq_np = model_np.fit(train_) ## Ver freq            
                
                future = model_np.make_future_dataframe(train_,periods=forecast_length) 
                forecast = model_np.predict(future)
                
                col="yhat1"
                y_pred=forecast.iloc[len(forecast)-forecast_length:,:]
                
            elif algo=='AutoArima':
                
                model_arima.fit(train_[[target]])
            
                forecast = model_arima.predict(n_periods=forecast_length)
                y_pred=pd.DataFrame(forecast, columns = [0]) #y_pred=forecast.to_frame()
                col=0
                
            y_pred=y_pred[col]
                        
            y_true=Test[target][0:forecast_length]
            
            y_pred_list = y_pred.tolist()
            y_true_list = y_true.tolist()
            
            """
            if barplots==True:
                plt.plot(y_true_list, label='Actual')
                plt.plot(y_pred_list, label='Predicted')
                plt.title(f"{algo} || Window Iteration: {rolling_cycle+1} of {iterations}.", loc='center')
                plt.legend()
                plt.show()
            """
            list_y_true.append(y_true)
            list_y_pred.append(y_pred)
            x=pd.concat(list_y_true)
            y=pd.concat(list_y_pred) 
            x=reset_index_DF(x)
            y=reset_index_DF(y)
            df = pd.DataFrame()
            df[['y_true']]=x
            df[['y_pred']]=y
            df[['Window']]=rolling_cycle
            df_=df.iloc[len(df)-forecast_length:,:]
            
            df_.index=Test['ds'][0:forecast_length]
            list_completa_dfs.append(df_)

    Df_Pred=pd.concat(list_completa_dfs)

    Df_Pred['Window']=Df_Pred['Window']+1

    return Df_Pred

def vertical_univariated_performance(Dataset:pd.DataFrame,
                                     forecast_length:int):
    """
    The vertical_univariated_performance function takes a dataframe and the forecast length as inputs.
    It returns a dataframe with the metrics for each step ahead, for all steps in the forecast_length.
    
    :param Dataset:pd.DataFrame: Pass the dataset that will be used to calculate the performance metrics
    :param forecast_length:int: Indicate the number of steps ahead that we want to forecast
    :return: A dataframe with the metrics of performance for each step ahead
    """
    df_=Dataset.copy()

    target="y"
    ahead_forecast_list,steps_list,list_dfs=list(range(1,forecast_length+1)),(list(dict.fromkeys(df_['Window'].tolist()))),[]

    for elemento_step in steps_list:
        df_Filtered=df_[df_['Window']==elemento_step]
        df_Filtered['V_Window']=ahead_forecast_list
        list_dfs.append(df_Filtered)
    Df_Vertical= pd.concat(list_dfs)
    
    ahead_steps,list_metrics = (list(dict.fromkeys(Df_Vertical['V_Window'].tolist()))),[]

    for element in ahead_steps:
        x=Df_Vertical.loc[Df_Vertical['V_Window'] == element]
        vertical_metrics=pd.DataFrame(metrics_regression(x['y_true'],x['y_pred']),index=[0])
        vertical_metrics[['Forecast_Length']]=element
        list_metrics.append(vertical_metrics)

    Total_Vertical_Metrics = pd.concat(list_metrics)

    return Total_Vertical_Metrics

def select_best_model(Dataset:pd.DataFrame,
                      eval_metric:str='MAE'): #'Mean Absolute Percentage Error','Mean Squared Error','Max Error'
    """
    The select_best_model function takes a Dataset containing the results of 
    several models and returns the best model based on an evaluation metric.
    
    
    :param Dataset:pd.DataFrame: Pass the dataframe to be evaluated
    :param eval_metric:str='MAE': Select the evaluation metric to be used in the function
    :return: The best model (the one with the lowest mean absolute error)
    """ 
    df=Dataset.copy()
    model_perf = {}
    
    if eval_metric == "MAE":
        eval_metric_='Mean Absolute Error'
    elif eval_metric == "MAPE":
        eval_metric_='Mean Absolute Percentage Error'
    elif eval_metric == "MSE":
        eval_metric_='Mean Squared Error'
    
    list_models=(list(dict.fromkeys(Dataset['Model'].tolist())))

    for model in list_models:
        
        df=Dataset.copy()
        df=df.loc[df['Model'] == model]
        metric_model=round(df[eval_metric_].mean(),4)
        
        model_perf[model] = metric_model
    print("Models Predictive Performance:", model_perf)
    best_model=min(model_perf, key=model_perf.get)
    perf_metric=model_perf[best_model]
    if len(list_models)>1:
        print("The model with best performance was", best_model, "with an (mean)", 
              eval_metric_, "of", perf_metric )
    return best_model

def pred_performance(Dataset:pd.DataFrame,
                     train_size:float,
                     forecast_size:int,
                     window_size:int,
                     list_models:list,
                     model_configs:dict,
                     granularity:str="1d",
                     eval_metric:str="MAE"):
    """
    The pred_performance function takes as input a dataframe, list of models and model configurations.
    It returns the best model based on the evaluation metric specified by the user (default is MAEs), 
    the performance metrics for all models, and predictions for each model.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be used for forecasting
    :param list_models:list: Specify the list of models that will be used for the prediction
    :param model_configs:dict: Pass the parameters of the model that is going to be used
    :param train_size:float: Define the size of the training set
    :param forecast_size:int: Define the forecast horizon
    :param window_size:int: Define the size of the rolling window used to train and predict
    :param granularity:str: Specify the time series granularity
    :param eval_metric:str=&quot;MAE&quot;: Select the best model based on the selected eval_metric
    :return: The best model (string), the performance of all models and the predictions for each model
    """
    Pred_Dfs,Pred_Values,target=[],[],"y"
    
    list_mv=['RandomForest','ExtraTrees','GBR','KNN','GeneralizedLR','AutoKeras','XGBoost','H2O_AutoML']
    list_uv=['NeuralProphet','Prophet','AutoArima']
    
    for elemento in list_models:
    
        if elemento in list_mv:

            results=Multivariate_Forecast(Dataset,
                                          train_length=train_size,
                                          forecast_length=forecast_size,
                                          rolling_window_size=window_size,
                                          algo=elemento,
                                          model_configs=model_configs,
                                          granularity=granularity)
            pred_perfomance=vertical_univariated_performance(results,forecast_size)
            pred_perfomance['Model']=elemento
            Pred_Dfs.append(pred_perfomance)
            results['Model']=elemento
            Pred_Values.append(results)
        
        if elemento in list_uv:
            results=Univariate_Forecast(Dataset,
                                        train_length=train_size,
                                        forecast_length=forecast_size,
                                        rolling_window_size=window_size,
                                        algo=elemento,
                                        model_configs=model_configs,
                                        granularity=granularity)
            pred_perfomance=vertical_univariated_performance(results,forecast_size)
            pred_perfomance['Model']=elemento
            Pred_Dfs.append(pred_perfomance)
            results['Model']=elemento
            Pred_Values.append(results)
            
        total_pred_results,predictions = pd.concat(Pred_Dfs),pd.concat(Pred_Values)
        total_pred_results,predictions=round_cols(total_pred_results, target),round_cols(predictions, target)
        
    Best_Model=select_best_model(total_pred_results,eval_metric=eval_metric)
        
    return Best_Model, total_pred_results, predictions

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
                 model_configs:dict,
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
    Dataset_Pred=pred_dataset(Dataset,forecast_size,granularity)
    Dataset["Values"]=0
    Dataset_Pred["Values"]=1
    Dataset_Final= pd.concat([Dataset,Dataset_Pred])
    Dataset_Final.index=Dataset_Final['Date']
    
    list_mv=['RandomForest','ExtraTrees','GBR','KNN','GeneralizedLR','AutoKeras','XGBoost','H2O_AutoML']
    list_uv=['NeuralProphet','Prophet','AutoArima']
    
    target='y'

    if selected_model in list_uv:
        
        Dataframe=Dataset.copy()
        Dataframe = Dataframe.rename(columns={'Date': 'ds'})
        Dataframe = Dataframe[['ds', target]]
        
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

        if selected_model =='NeuralProphet':
            np_params=model_configs['NeuralProphet']
            model_np = NeuralProphet(**np_params)
            freq_np = model_np.fit(Dataframe)           
            
            future = model_np.make_future_dataframe(Dataframe,periods=forecast_size) 
            forecast = model_np.predict(future)
            forecast.head()
            col='yhat1'
            y_pred=forecast.iloc[len(forecast)-forecast_size:,:]
            
        elif selected_model =='Prophet':
            pr_params=model_configs['Prophet']
            m = Prophet(**pr_params)
            model_p = m.fit(Dataframe)
            
            future = model_p.make_future_dataframe(periods=forecast_size,freq=frequency)
            forecast = model_p.predict(future)
            col='yhat'
            y_pred=forecast.iloc[len(forecast)-forecast_size:,:]
            
        elif selected_model =='AutoArima':
            aa_params=model_configs['AutoArima']
            model_arima=auto_arima(Dataframe[['y']], **aa_params)
            y_pred = model_arima.predict(n_periods=forecast_size)
            y_pred=pd.DataFrame(y_pred, columns = [0])
            col=0
        
        y_pred=y_pred[col]
        
        Dataset_Final['y'][len(Dataset_Final)-forecast_size:]=y_pred
        Dataset_Final['Values']=Dataset_Final['Values'].replace(0,'Real')
        Dataset_Final['Values']=Dataset_Final['Values'].replace(1,'Predicted')
        Dataset_Final=round_cols(Dataset_Final, target)

    elif selected_model in list_mv:

        Dataset_Final=engin_date(Dataset_Final,Drop=False)
        Dataset_Final=multivariable_lag(Dataset_Final,target,range_lags=[forecast_size,forecast_size*3]) 
        
        Train=Dataset_Final.iloc[:len(Dataset_Final)-forecast_size,:]
        Test=Dataset_Final.iloc[len(Dataset_Final)-forecast_size:,:]

        input_cols=list(Dataset_Final.columns)
        input_cols.remove(target)  
        
        scaler = MinMaxScaler()
        
        scaler = scaler.fit(Train[input_cols])
        Train[input_cols] = scaler.transform(Train[input_cols])
        Test[input_cols] = scaler.transform(Test[input_cols])

        y_pred=model_prediction(Train, Test,target,model_configs=model_configs,algo=selected_model)
        Dataset_Final['y'][len(Dataset_Final)-forecast_size:]=y_pred
        Dataset_Final['Values']=Dataset_Final['Values'].replace(0,'Real')
        Dataset_Final['Values']=Dataset_Final['Values'].replace(1,'Predicted')
        Dataset_Final['Date']=Dataset_Final.index
        Dataset_Final=Dataset_Final[['Date','y','Values']]

    return Dataset_Final
