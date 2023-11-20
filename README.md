<br>
<p align="center">
  <h2 align="center"> TSForecasting - Automated Time Series Forecasting Framework
  <br>
  
## Framework Contextualization <a name = "ta"></a>

The `TSForecasting` project constitutes an complete and integrated pipeline to Automate Time Series Forecasting applications through the implementation of multivariate approaches integrating regression models referring to modules such as `SKLearn`, `H2O.ai`, `XGBoost` and also univariate approaches of more classics methods such as `Prophet` and `AutoArima`, this following an 'Expanding Window' performance evaluation.

The architecture design includes five main sections, these being: data preprocessing, feature engineering, hyperparameter optimization, forecast ensembling and forecasting method selection which are organized and customizable in a pipeline structure.

This project aims at providing the following application capabilities:

* General applicability on tabular datasets: The developed forecasting procedures are applicable on any data table associated with any Time Series Forecasting scopes.

* Hyperparameter optimization and customization: It provides full configuration for each model hyperparameter through the customization of `model_configurations` dictionary, allowing optimal performance to be obtained for any use case.
    
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
sliding size is equal to the forecast size, each next segment starts at the end of the previous.
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
    
The first needed step after importing the package is to load a dataset and define your DataTime (`datetime64[ns]` type) and Target column to be predicted, then rename them to `Date` and `y`, respectively.
The following step is to define your future running pipeline parameters variables, these being:
* train_size: Length of Train data in which will be applied the first Expanding Window iteration;  
* forecast_size: Full length of test/future ahead predictions;
* sliding_size: Length of sliding window, sliding_size>=forecast_size is suggested;
* models: Select all the models intented to ensemble to evaluation. To fit and compare predictive performance of available models set them in paramater `models:list`, options are the following:
  * `RandomForest`
  * `ExtraTrees`
  * `GBR`
  * `KNN`
  * `GeneralizedLR`
  * `XGBoost`
  * `H2O_AutoML`
  * `AutoArima`
  * `Prophet`
* hparameters: Nested dictionary in which are contained all models and specific hyperparameters configurations. Feel free to customize each model as you see fit (customization example shown bellow); 
* granularity: Valid interval of periods correlated to data -> 1m,30m,1h,1d,1wk,1mo (default='1d');
* metric: Default predictive evaluation metric is `MAE` (Mean Absolute Error), other options are `MAPE` (Mean Absolute Percentage Error) and `MSE`
(Mean Squared Error);
 
The `fit_forecast` method set the default parameters for fitting and comparison of all segmented windows for each selected and configurated model. After implementation, the `history` method agregates the returning variables `fit_performance` containing every detailed measure of each `window` iteration predicted value and `fit_predictions` measuring all segmented `window` iterations performance.

The `forecast` method forecasts the future values based on the previously predefined best performing model.
        
```py

from tsforecasting.forecasting import TSForecasting
from tsforecasting.parameters import model_configurations
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console
import h2o

h2o.init() # -> Run only if using H2O_AutoML models   

## Dataframe Loading
data = pd.read_csv('csv_directory_path') 
data = data.rename(columns={'DateTime_Column': 'Date','Target_Name_Column':'y'})
data = data[['Date',"y"]]
    
## Get Models Hyperparameters Configurations
parameters = model_configurations()
print(parameters)

# Customization Hyperparameters Settings
parameters["RandomForest"]["n_estimators"] = 200
parameters["KNN"]["n_neighbors"] = 5
parameters["Prophet"]["seasonality_mode"] = 'multiplicative'
parameters["H2O_AutoML"]["max_runtime_secs"] = 90

## Fit Forecasting Evaluation
tsf = TSForecasting(train_size = 0.95,
                    forecast_size = 15,
                    sliding_size = 15,
                    models = ['RandomForest','ExtraTrees', 'GBR', 'KNN', 'GeneralizedLR',
                              'XGBoost', 'AutoArima','Prophet','H2O_AutoML'],
                    hparameters = parameters,
                    granularity = "1h", # 1m,30m,1h,1d,1wk,1mo
                    metric = "MAE"      # MAPE, MSE
                    )
tsf = tsf.fit_forecast(dataset = data)

# Get Fit History
fit_predictions, fit_performance = tsf.history()

## Forecast
forecast = tsf.forecast()

```  

## 2. TSForecasting - Extra Auxiliar Methods
    
The `engin_date` method converts and transforms columns of Datetime type into additional columns (Year, Day of the  Year, Season, Month, Day of the month, Day of the week, Weekend, Hour, Minute, Second) which will be added by association to the input dataset and subsequently deletes the original column if parameter `drop`=`True`.

The `multivariable_lag` method creats all the past lags related to the target `y` feature automatically (in accordance to `range_lags` parameter) and adds each constructed column into the dataset.
 
```py   

# Feature Engineering 

from TSForecasting_imports.treatment import Treatment

tr = Treatment()

data = tr.engin_date(dataset = data,
                     drop = False) 

data = tr.multivariable_lag(dataset = data,
                            range_lags = [1,10],
                            drop_na = True)    
```
    
## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/TSForecasting/blob/main/LICENSE) for more information.

## Contact 
 
Luis Santos - [LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)
