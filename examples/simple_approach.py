from tsforecasting import * # Importing Model_Configs directly from source package
import tsforecasting as tsf
import pandas as pd
import h2o
import warnings
warnings.filterwarnings("ignore",category=Warning)

#h2o.init() # -> Run only if using H2O_AutoML models   

#source_data="https://www.kaggle.com/datasets/kandij/electric-production"

data = pd.read_csv('TSForecasting_dev/data/Electric_Production.csv')

data['DATE'] = pd.to_datetime(data['DATE'])

data = data.rename(columns={'DATE': 'Date','Value':'y'})
data=data[['Date',"y"]]

## Forecast Customization
train_size_=0.9
forecast_size_=10
window_size_=10
granularity_='1d' # 1m,30m,1h,1d,1wk,1mo
list_models_=['RandomForest', 'ExtraTrees', 'GBR', 'KNN', 'GeneralizedLR', 'XGBoost','AutoArima','Prophet']#, 'AutoKeras',#H2O_AutoML 'Prophet']
eval_metric_="MAE" # MAPE, MSE


## Customizing parameters settings
Model_Configs["RandomForest"]["n_estimators"]=50
Model_Configs["ExtraTrees"]["n_estimators"]=50
Model_Configs["GBR"]["n_estimators"]=50


best_model,perf_results,predictions=tsf.pred_performance(Dataset=data, 
                                                         train_size=train_size_,
                                                         forecast_size=forecast_size_,
                                                         window_size=window_size_,
                                                         list_models=list_models_,
                                                         model_configs=Model_Configs,
                                                         granularity=granularity_,
                                                         eval_metric=eval_metric_)

dataset_pred=tsf.pred_results(Dataset=data,
                              forecast_size=forecast_size_,
                              model_configs=Model_Configs,
                              granularity=granularity_,
                              selected_model=best_model)

