from tsforecasting.forecasting import TSForecasting
from tsforecasting.parameters import model_configurations
import pandas as pd
import h2o
import warnings
warnings.filterwarnings("ignore",category=Warning)

h2o.init() # -> Run only if using H2O_AutoML models   

#source_data="https://www.kaggle.com/datasets/kandij/electric-production"

url="https://raw.githubusercontent.com/TsLu1s/TSForecasting/main/data/Electric_Production.csv"

## Dataframe Loading
data= pd.read_csv(url)
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.rename(columns={'DATE': 'Date','Value':'y'})
data=data[['Date',"y"]]

## Get models hyperparameters configurations
parameters = model_configurations()
print(parameters)

# Customization Hyperparameters Settings
parameters["RandomForest"]["n_estimators"] = 200
parameters["KNN"]["n_neighbors"] = 5
parameters["H2O_AutoML"]["max_runtime_secs"] = 45

## Fit Forecasting Evaluation
tsf = TSForecasting(train_size = 0.865,
                    forecast_size = 10,
                    sliding_size = 10,
                    models = ['RandomForest','ExtraTrees', 'GBR', 'KNN', 'GeneralizedLR',
                              'XGBoost', 'AutoArima','Prophet','H2O_AutoML'],
                    hparameters = parameters,
                    granularity = "1mo", # 1m,30m,1h,1d,1wk,1mo
                    metric = "MAE"      # MAPE, MSE
                    )

tsf = tsf.fit_forecast(dataset = data)

# Get Fit History
fit_predictions, fit_performance = tsf.history()

## Forecast
forecast = tsf.forecast()

