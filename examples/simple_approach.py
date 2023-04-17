import tsforecasting as tsf
import pandas as pd
import h2o
import warnings
warnings.filterwarnings("ignore",category=Warning)

#h2o.init() # -> Run only if using H2O_AutoML models   

#source_data="https://www.kaggle.com/datasets/kandij/electric-production"

url="https://raw.githubusercontent.com/TsLu1s/TSForecasting/main/data/Electric_Production.csv"

data= pd.read_csv(url)
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.rename(columns={'DATE': 'Date','Value':'y'})
data=data[['Date',"y"]]


## Forecast Customization
train_size_=0.9
forecast_size_=10
window_size_=10
granularity_='1d' # 1m,30m,1h,1d,1wk,1mo
list_models_=['RandomForest', 'ExtraTrees', 'GBR', 'KNN', 'XGBoost','AutoArima','Prophet'] #,'H2O_AutoML'] 
eval_metric_="MAE" # MAPE, MSE


## Customizing hparameters settings
models_hparameters=tsf.model_configurations()
models_hparameters["RandomForest"]["n_estimators"]=100
models_hparameters["ExtraTrees"]["n_estimators"]=100
models_hparameters["GBR"]["n_estimators"]=100


## Forecast Model Ensemble Evaluation
best_model,perf_results,predictions=tsf.pred_performance(dataset=data,
                                                         train_size=train_size_,
                                                         forecast_size=forecast_size_,
                                                         window_size=window_size_,
                                                         list_models=list_models_,
                                                         model_configs=models_hparameters,
                                                         granularity=granularity_,
                                                         eval_metric=eval_metric_)


## Forecast with Best Performing Model
dataset_pred=tsf.pred_results(dataset=data,
                              forecast_size=forecast_size_,
                              model_configs=models_hparameters,
                              granularity=granularity_,
                              selected_model=best_model)

