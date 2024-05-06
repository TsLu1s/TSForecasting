from tsforecasting.forecasting import (TSForecasting, 
                                       model_configurations)
import pandas as pd
import warnings
warnings.filterwarnings("ignore",category=Warning)

#source_data="https://www.kaggle.com/datasets/kandij/electric-production"

url="https://raw.githubusercontent.com/TsLu1s/TSForecasting/main/data/Electric_Production.csv"

## Dataframe Loading
data= pd.read_csv(url)
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.rename(columns={'DATE': 'Date','Value':'y'})
data=data[['Date',"y"]]

## Get models hyperparameters configurations
hparameters = model_configurations()
print(hparameters)

# Customization Hyperparameters Settings
hparameters["RandomForest"]["n_estimators"] = 20
hparameters["KNN"]["n_neighbors"] = 5
hparameters["Catboost"]["iterations"] = 150
hparameters["AutoGluon"]["time_limit"] = 50

#from forecasting import TSForecasting

## Fit Forecasting Evaluation
tsf = TSForecasting(train_size = 0.865,
                    lags = 10,
                    horizon = 5,
                    sliding_size = 10,
                    models = ['RandomForest', 'GeneralizedLR', 'GBR', 'KNN', 'GeneralizedLR',
                              'XGBoost', 'LightGBM', 'Catboost', 'AutoGluon'],
                    hparameters = hparameters,
                    granularity = '1mo',
                    metric = 'MAE'
                   )

tsf.fit_forecast(dataset = data)

# Get Fit History
fit_performance = tsf.history()

## Forecast
forecast = tsf.forecast()
