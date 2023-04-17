import pandas as pd
import numpy as np
from sklearn.metrics import *
from .tsf_data_eng import round_cols
from .tsf_expanding_window import Univariate_Forecast, Multivariate_Forecast
from .tsf_model_configs import model_configurations

h_parameters=model_configurations()

def metrics_regression(y_true, y_prev):
    """
    The metrics_regression function calculates the metrics for regression models.
    It takes as input two arrays: y_true and y_prev, which are the real values 
    of a target variable and its predictions respectively. The function returns 
    a dictionary with all the metrics.
    
    :param y_true: Store the real values of the target variable
    :param y_prev: Compare the real values of y with the predicted values
    :return: A dictionary with the metrics of the regression model
    """
    mae=mean_absolute_error(y_true, y_prev)
    mape = (mean_absolute_percentage_error(y_true, y_prev))*100
    mse=mean_squared_error(y_true, y_prev)
    evs= explained_variance_score(y_true, y_prev)
    maximo_error= max_error(y_true, y_prev)
    metrics_reg = {'Mean Absolute Error': mae, 
                   'Mean Absolute Percentage Error': mape,
                   'Mean Squared Error': mse,
                   'Explained Variance Score': evs, 
                   'Max Error': maximo_error}
    
    return metrics_reg

def vertical_univariated_performance(dataset:pd.DataFrame,
                                     forecast_length:int):
    """
    The vertical_univariated_performance function takes a dataframe and the forecast length as inputs.
    It returns a dataframe with the metrics for each step ahead, for all steps in the forecast_length.
    
    :param dataset:pd.DataFrame: Pass the dataset that will be used to calculate the performance metrics
    :param forecast_length:int: Indicate the number of steps ahead that we want to forecast
    :return: A dataframe with the metrics of performance for each step ahead
    """
    df_=dataset.copy()

    target="y"
    ahead_forecast_list,steps_list,list_dfs=list(range(1,forecast_length+1)),(list(dict.fromkeys(df_['Window'].tolist()))),[]

    for elemento_step in steps_list:
        df_filter=df_[df_['Window']==elemento_step]
        df_filter['V_Window']=ahead_forecast_list
        list_dfs.append(df_filter)
    df_vertical= pd.concat(list_dfs)
    
    ahead_steps,list_metrics = (list(dict.fromkeys(df_vertical['V_Window'].tolist()))),[]

    for element in ahead_steps:
        x=df_vertical.loc[df_vertical['V_Window'] == element]
        vertical_metrics=pd.DataFrame(metrics_regression(x['y_true'],x['y_pred']),index=[0])
        vertical_metrics[['Forecast_Length']]=element
        list_metrics.append(vertical_metrics)

    total_vertical_metrics = pd.concat(list_metrics)

    return total_vertical_metrics

def select_best_model(dataset:pd.DataFrame,
                      eval_metric:str='MAE'): #'Mean Absolute Percentage Error','Mean Squared Error','Max Error'
    """
    The select_best_model function takes a dataset containing the results of 
    several models and returns the best model based on an evaluation metric.
    
    
    :param dataset:pd.DataFrame: Pass the dataframe to be evaluated
    :param eval_metric:str='MAE': Select the evaluation metric to be used in the function
    :return: The best model (the one with the lowest mean absolute error)
    """ 
    df=dataset.copy()
    model_perf = {}
    
    if eval_metric == "MAE":
        eval_metric_='Mean Absolute Error'
    elif eval_metric == "MAPE":
        eval_metric_='Mean Absolute Percentage Error'
    elif eval_metric == "MSE":
        eval_metric_='Mean Squared Error'
    
    list_models=(list(dict.fromkeys(dataset['Model'].tolist())))

    for model in list_models:
        
        df=dataset.copy()
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

def pred_performance(dataset:pd.DataFrame,
                     train_size:float,
                     forecast_size:int,
                     window_size:int,
                     list_models:list,
                     model_configs:dict=h_parameters,
                     granularity:str="1d",
                     eval_metric:str="MAE"):
    """
    The pred_performance function takes as input a dataframe, list of models and model configurations.
    It returns the best model based on the evaluation metric specified by the user (default is MAEs), 
    the performance metrics for all models, and predictions for each model.
    
    :param dataset:pd.DataFrame: Pass the dataset to be used for forecasting
    :param list_models:list: Specify the list of models that will be used for the prediction
    :param model_configs:dict: Pass the parameters of the model that is going to be used
    :param train_size:float: Define the size of the training set
    :param forecast_size:int: Define the forecast horizon
    :param window_size:int: Define the size of the rolling window used to train and predict
    :param granularity:str: Specify the time series granularity
    :param eval_metric:str=&quot;MAE&quot;: Select the best model based on the selected eval_metric
    :return: The best model (string), the performance of all models and the predictions for each model
    """
    pred_dfs,pred_values,target=[],[],"y"
    
    list_mv=['RandomForest','ExtraTrees','GBR','KNN','GeneralizedLR','XGBoost','H2O_AutoML']
    list_uv=['Prophet','AutoArima']
    
    for elemento in list_models:
    
        if elemento in list_mv:

            results=Multivariate_Forecast(dataset,
                                          train_length=train_size,
                                          forecast_length=forecast_size,
                                          rolling_window_size=window_size,
                                          algo=elemento,
                                          model_configs=model_configs,
                                          granularity=granularity)
            pred_perfomance=vertical_univariated_performance(results,forecast_size)
            pred_perfomance['Model']=elemento
            pred_dfs.append(pred_perfomance)
            results['Model']=elemento
            pred_values.append(results)
        
        if elemento in list_uv:
            results=Univariate_Forecast(dataset,
                                        train_length=train_size,
                                        forecast_length=forecast_size,
                                        rolling_window_size=window_size,
                                        algo=elemento,
                                        model_configs=model_configs,
                                        granularity=granularity)
            pred_perfomance=vertical_univariated_performance(results,forecast_size)
            pred_perfomance['Model']=elemento
            pred_dfs.append(pred_perfomance)
            results['Model']=elemento
            pred_values.append(results)
            
        total_pred_results,predictions = pd.concat(pred_dfs),pd.concat(pred_values)
        total_pred_results,predictions=round_cols(total_pred_results, target),round_cols(predictions, target)
        
    best_model=select_best_model(total_pred_results,eval_metric=eval_metric)
        
    return best_model, total_pred_results, predictions
