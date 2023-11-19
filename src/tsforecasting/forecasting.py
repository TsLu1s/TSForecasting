import pandas as pd
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from pmdarima.arima import auto_arima
from tsforecasting.parameters import model_configurations
from tsforecasting.variational import Variational_Forecast
from tsforecasting.evaluation import vertical_performance, select_model

class TSForecasting(Variational_Forecast):
    def __init__(self,
                 train_size: float = 0.8,
                 forecast_size: int = 15, 
                 sliding_size: int = 15,
                 models: list = ['RandomForest','GBR','XGBoost','AutoArima'],
                 hparameters: dict = model_configurations(),
                 granularity: str = '1d',
                 metric: str = 'MAE'):
        super().__init__(dataset = pd.DataFrame(),
                         train_size = train_size,
                         forecast_size = forecast_size,
                         sliding_size = sliding_size,
                         hparameters = hparameters,
                         granularity = granularity)  
        self.models = models
        self.metric = metric
        self.target = 'y'
        self.fit_performance = None
        self.fit_predictions = None
        self.selected_model = None
        
    def fit_forecast(self,
                     dataset:pd.DataFrame):
        """
        Fit forecasting models for each specified model and evaluate their performance.

        Args:
            dataset (pd.DataFrame): Time series dataset.
        """
    
        self.dataset,metrics,forecasts=dataset,[],[]
        
        list_mv=['RandomForest','ExtraTrees','GBR','KNN','GeneralizedLR','XGBoost','H2O_AutoML']
        list_uv=['Prophet','AutoArima']
        
        for model in self.models:
            if model in list_mv: results=super().multivariate_forecast(algo=model) 
            if model in list_uv: results=super().univariate_forecast(algo=model)
            results['Model']=model
            perf=vertical_performance(results,self.forecast_size)
            perf['Model']=model
            forecasts.append(results),metrics.append(perf)
                
            self.fit_performance,self.fit_predictions=pd.concat(metrics).reset_index(drop=True),pd.concat(forecasts)
            
        self.selected_model=select_model(self.fit_performance,metric=self.metric)
        
        return self
    
    def history(self):
        """
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Fit predictions and performance metrics.
        """
        return self.fit_predictions, self.fit_performance
    
    def forecast(self):
    
        X=self.dataset.copy()
        X=super().slice_timestamp(X)
        X_=super().future_timestamps(X,
                                     self.forecast_size,
                                     self.granularity)
        
        # Set marker values for original and forecasted data
        X['Values']=0
        X_['Values']=1
        X_=pd.concat([X,X_])
        X_.index=X_['Date']
        
        list_mv=['RandomForest','ExtraTrees','GBR','KNN','GeneralizedLR','XGBoost','H2O_AutoML']
        list_uv=['Prophet','AutoArima']
        me1,me2='Mean Absolute Error','Max Error'
        
        # Check if the selected model is univariate or multivariate
        if self.selected_model in list_uv:
            
            df=X.copy()
            df=df.rename(columns={'Date':'ds'})
            df=df[['ds', self.target]]
            
            # Determine the frequency based on the granularity
            if self.granularity=='1m':
                frequency='min'
            elif self.granularity=='30m':
                frequency='30min'
            elif self.granularity=='1h':
                frequency='H'
            elif self.granularity=='1d':
                frequency='D'
            elif self.granularity=='1wk':
                frequency='W'
            elif self.granularity=='1mo':
                frequency='M'
                
            if self.selected_model == 'Prophet':
                pr_params=self.hparameters['Prophet']
                m=Prophet(**pr_params)
                model_p=m.fit(df)
                
                future=model_p.make_future_dataframe(periods=self.forecast_size,freq=frequency)
                forecast=model_p.predict(future)
                col='yhat'
                y_pred=forecast.iloc[len(forecast)-self.forecast_size:,:]
                
            elif self.selected_model == 'AutoArima':
                aa_params=self.hparameters['AutoArima']
                model_arima=auto_arima(df[[self.target]], **aa_params)
                y_pred=model_arima.predict(n_periods=self.forecast_size)
                y_pred=pd.DataFrame(y_pred, columns=[0])
                col=0
            y_pred=y_pred[col]
            
        
        elif self.selected_model in list_mv:
            
            X_=super().engin_date(X_,drop=False)
            X_=super().multivariable_lag(X_,
                                         range_lags=[self.forecast_size,self.forecast_size*self.x_lag],
                                         drop_na=True)
            
            train=X_.iloc[:len(X_)-self.forecast_size,:]
            test=X_.iloc[len(X_)-self.forecast_size:,:]
        
            input_cols=list(train.columns)
            input_cols.remove(self.target)
            
            # Standardize features using StandardScaler
            scaler=StandardScaler()
            scaler=scaler.fit(train[input_cols])
            
            train[input_cols]=scaler.transform(train[input_cols])
            test[input_cols]=scaler.transform(test[input_cols])
            
            # Perform prediction based on the selected model
            if self.selected_model=='H2O_AutoML':
                y_pred=super().prediction(train=train,
                                          test=test,
                                          algo=self.selected_model,
                                          id_h2o_model=self.id_h2o_leader)
            else: y_pred=super().prediction(train=train,
                                            test=test,
                                            algo=self.selected_model)
        
        # Update the forecasted values in the dataset
        X_[self.target][len(X_)-self.forecast_size:]=y_pred
        X_ = X_[-self.forecast_size:]
        X_['Date']=X_.index
        X_=X_.reset_index(drop=True)
        
        # Calculate upper and lower confidence intervals
        X_['y_superior'], X_['y_inferior'] = X_['y'].add(self.fit_performance[me1]), X_['y'].sub(self.fit_performance[me1])
        X_['y_max_interval'], X_['y_min_interval'] = X_['y'].add(self.fit_performance[me2]), X_['y'].sub(self.fit_performance[me2])
        X_=X_[['Date',self.target,'y_superior','y_inferior','y_max_interval','y_min_interval']]

        return X_
