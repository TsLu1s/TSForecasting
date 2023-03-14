<br>
<p align="center">
  <h2 align="center"> TSForecasting - Automated Time Series Forecasting Framework
  <br>
  
## Framework Contextualization <a name = "ta"></a>

The `TSForecasting` project constitutes an complete and integrated pipeline to Automate Time Series Forecasting applications through the implementation of multivariate approaches integrating regression models referring to modules such as SKLearn, H2O.ai, XGBoost and also univariate approaches of more classics methods such as Prophet and AutoArima, this following an 'Expanding Window' performance evaluation.

The architecture design includes five main sections, these being: data preprocessing, feature engineering, hyperparameter optimization, forecast ensembling and forecasting method selection which are organized and customizable in a pipeline structure.

This project aims at providing the following application capabilities:

* General applicability on tabular datasets: The developed forecasting procedures are applicable on any data table associated with any Time Series Forecasting scopes, based on DateTime and Target columns to be predicted.

* Hyperparameter optimization and customization: It provides full configuration for each model hyperparameter through the customization of `model_configs` parameter variable values, allowing optimal performance to be obtained for each use case.
    
* Robustness and improvement of predictive results: The implementation of the TSForecasting pipeline aims to improve the predictive performance directly associated with the application of the best performing forecasting method. 
   
#### Main Development Tools <a name = "pre1"></a>

Major frameworks used to built this project: 

* [Sklearn](https://scikit-learn.org/stable/)
* [H2O.ai](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/)
* [AutoArima](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)
* [Prophet](https://facebook.github.io/prophet/docs/quick_start.html#python-api)
    
## Performance Evaluation Structure <a name = "ta"></a>

<p align="center">
  <img src="https://i.ibb.co/ctYj6tt/Expanding-Window-TSF.png" align="center" width="450" height="350" />
  
</p>  
    
The Expanding Window evaluation technique is a temporary approximation on the real value of the time series data. 
The first test segment is selected according to the train length and then it's forecasted in accordance with forecast size.
The starting position of the subsequent segment is set in direct relation to the sliding window size, this meaning, if the
window size is equal to the forecast size, each next segment starts at the end of the previous.
This process is repeated until all time series data gets segmented and it uses all the iterations and observations
to construct an aggregated and robust performance analysis to each predicted point.

## Where to get it <a name = "ta"></a>

Binary installer for the latest released version is available at the Python Package Index (PyPI).   

## Installation  

To install this package from Pypi repository run the following command:

```
pip install tsforecasting
```

# Usage Examples
    
## 1. TSForecasting - Automated Time Series Forecasting
    
The first needed step after importing the package is to load a dataset and define your DataTime and to be predicted Target column and rename them to 'Date' and 'y', respectively.
The following step is to define your future running pipeline parameters variables, this being:
* train_size: Length of Train data in which will be applied the first Expanding Window iteration;  
* forecast_size: Full length of test/future ahead predictions;
* window_size: Length of sliding window, Window_Size>=Forecast_Size is recommended;
* granularity: Valid interval of periods correlated to data -> 1m,30m,1h,1d,1wk,1mo (default='1d');
* eval_metric: Default predictive evaluation metric (eval_metric) is "MAE" (Mean Absolute Error), other options are "MAPE" (Mean Absolute Percentage Error) and "MSE"
(Mean Squared Error);
* list_models: Select all the models intented do run in `pred_performance` function. To compare predictive performance of all available models set paramater `list_models`=['RandomForest','ExtraTrees','GBR',
'KNN','GeneralizedLR','XGBoost','H2O_AutoML','AutoArima','Prophet'];
* model_configs: Nested dictionary in which are contained all models and specific hyperparameters configurations. Feel free to customize each model as you see fit (customization example shown bellow); 
 
The `pred_performance` function compares all segmented windows values (predicted and real) for each selected and configurated model then calculates it's predicted performance error metrics, returning the variable `best_model`(most effective model), `perf_results`[DataFrame] containing every detailed measure of each Test predicted value and at last the variable `predictions`[DataFrame] containing every segmented window iteration performed which can be use for analysis and objective models comparison. 

The `pred_results` function forecasts the future values based on the previously predefined parameters and the `selected model` which specifies the choosen model used to obtain future predictions.
    
Importante Note:

* Although not advisable to forecast without evaluating predictive performance first, forecast can be done without using the `pred_performance` evaluation function, by replacing the `selected_model` parameter (default='RandomForest') in the `pred_results` function with any choosen model.

    
```py

import tsforecasting as tsf
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console
import h2o

h2o.init() # -> Run only if using H2O_AutoML models   

data = pd.read_csv('csv_directory_path') # Dataframe Loading Example
    
data = data.rename(columns={'DateTime_Column': 'Date','Target_Name_Column':'y'})
data=data[['Date',"y"]]
    
train_size_=0.95
forecast_size_=15
window_size_=forecast_size_ # Recommended
granularity_='1d' # 1m,30m,1h,1d,1wk,1mo
eval_metric_="MAE" # MAPE, MSE
list_models_=['RandomForest','ExtraTrees','KNN','XGBoost','AutoArima'] # Ensemble Example
    
## Get models hyperparameters configurations
    
models_hparameters=tsf.model_configurations()
print(models_hparameters)

# Customization Example
models_hparameters["RandomForest"]["n_estimators"]=100
models_hparameters["ExtraTrees"]["n_estimators"]=100

## Performance Evaluation
best_model,perf_results,predictions=tsf.pred_performance(Dataset=data,
                                                         train_size=train_size_,
                                                         forecast_size=forecast_size_,
                                                         window_size=window_size_,
                                                         list_models=list_models_,
                                                         model_configs=models_hparameters,
                                                         granularity=granularity_,
                                                         eval_metric=eval_metric_)
## Forecast
dataset_pred=tsf.pred_results(Dataset=data,
                              forecast_size=forecast_size_,
                              model_configs=models_hparameters,
                              granularity=granularity_,
                              selected_model=best_model)
```  

## 2. TSForecasting - Extra Auxiliar Functions

The `model_prediction` function predicts your Test target column based on the input DataFrames, Train and Test, model configuration set by the parameter `model_configs` and the selected running algorithm in the parameter `algo` (default='RandomForest'). Note, you can select and customize any of the 11 models available in `Model_Configs` dictionary.
    
```py     
 
# Automated Model Predictions
 
y_predict = tsf.model_prediction(Train:pd.DataFrame,
                                 Test:pd.DataFrame,
                                 target:str="y",
                                 model_configs:dict=models_hparameters,
                                 algo:str='RandomForest')
```       
    
    
The `engin_date` function converts and transforms columns of Datetime type into additional columns (Year, Day of the  Year, Season, Month, Day of the month, Day of the week, Weekend, Hour, Minute) which will be added by association to the input dataset and subsequently deletes the original column if variable Drop=True.

The `multivariable_lag` function creats all the past lags automatically (in accordance to `range_lags` parameter) and adds each column into the input DataFrame.
 
```py   

# Feature Engineering 
    
dataset = tsf.engin_date(Dataset:pd.DataFrame,
                         Drop:bool=False) 

dataset = tsf.multivariable_lag(Dataset:pd.DataFrame,
                                target:str="y",
                                range_lags:list=[1,10],
                                drop_na:bool=True)
    
```

This `feature_selection_tb` function filters the most valuable features from the dataset. It's based on calculated variable importance in tree based regression models from Scikit-Learn and it can be customized by use of the parameter `total_vi` (total sum of relative variable\feature importance percentage selected) and `algo` selecting the model for evaluation ('ExtraTrees','RandomForest' and 'GBR').

```py  

# Feature Selection 

selected_columns, selected_importance_df=tsf.feature_selection_tb(Dataset:pd.DataFrame,
                                                                  target:str="y",
                                                                  total_vi:float=0.99,
                                                                  algo:str="ExtraTrees",
                                                                  estimators:int=250)
 ```   
    
You can analyse the obtained performance results by using the `metrics_regression` function wich contains the most used metrics for regression predictive contexts.
    
```py  
 
# Regression Performance Metrics

reg_performance = pd.DataFrame(tsf.metrics_regression(y_true,y_pred),index=[0])    # y_true:list, y_pred:list
        
```
    
## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/TSForecasting/blob/main/LICENSE) for more information.

## Contact 
 
Luis Santos - [LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)
    
Feel free to contact me and share your feedback.
