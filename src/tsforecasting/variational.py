import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ( 
    RandomForestRegressor, 
    ExtraTreesRegressor, 
    GradientBoostingRegressor
    )
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import TweedieRegressor
import xgboost
import h2o
from h2o.automl import H2OAutoML
from prophet import Prophet
from pmdarima.arima import auto_arima
from tsforecasting.treatment import Treatment

class Variational_Forecast(Treatment): 
    def __init__(self,
                 dataset: pd.DataFrame,
                 train_size: float,
                 forecast_size: int,
                 sliding_size: int,
                 hparameters: dict = {},
                 granularity: str = '1d'):
        self.dataset = dataset
        self.target = 'y'
        self.train_size = train_size
        self.forecast_size = forecast_size
        self.sliding_size = sliding_size
        self.hparameters = hparameters
        self.granularity = granularity
        self.id_h2o_leader = None
        self.x_lag = 3
        self.h2o_leaderboard=None
        
    def prediction(self,
                   train: pd.DataFrame,
                   test: pd.DataFrame,
                   algo: str = 'RandomForest',
                   id_h2o_model: str = None):
        """
        Generates predictions using specified and user customized regression algorithms.

        Args:
            train (pd.DataFrame): Training data with features and target.
            test (pd.DataFrame): Testing data with features and target.
            algo (str, optional): Regression algorithm to use ('RandomForest', 'ExtraTrees', 'GBR', 'KNN', 'GeneralizedLR',
                                  'H2O_AutoML', 'XGBoost'). Defaults to 'RandomForest'.
            id_h2o_model (str, optional): Model ID if exists previous use of H2O_AutoML training. Defaults to None.

        Returns:
            array-like: Predicted values.
        """
        train,test=super().slice_timestamp(train),super().slice_timestamp(test)
        
        X_train=train.iloc[:, 0:(len(list(train.columns))-1)].values
        X_test=test.iloc[:, 0:(len(list(train.columns))-1)].values
        y_train=train.iloc[:, (len(list(test.columns))-1)].values
        
        if algo == 'RandomForest' :
            rf_params=self.hparameters['RandomForest']
            regressor_RF=RandomForestRegressor(**rf_params)
            regressor_RF.fit(X_train, y_train)
            y_predict=regressor_RF.predict(X_test)
            
        elif algo == 'ExtraTrees' :
            et_params=self.hparameters['ExtraTrees']
            regressor_ET=ExtraTreesRegressor(**et_params)
            regressor_ET.fit(X_train, y_train)
            y_predict=regressor_ET.predict(X_test)
            
        elif algo == 'GBR' :
            gbr_params=self.hparameters['GBR']
            regressor_GBR=GradientBoostingRegressor(**gbr_params)
            regressor_GBR.fit(X_train, y_train)
            y_predict=regressor_GBR.predict(X_test)
            
        elif algo == 'KNN' :
            knn_params=self.hparameters['KNN']
            regressor_KNN=KNeighborsRegressor(**knn_params)
            regressor_KNN.fit(X_train, y_train)
            y_predict=regressor_KNN.predict(X_test)
            
        elif algo == 'GeneralizedLR' :
            td_params=self.hparameters['GeneralizedLR']
            regressor_TD=TweedieRegressor(**td_params)
            regressor_TD.fit(X_train, y_train)
            y_predict=regressor_TD.predict(X_test)
            
        elif algo =='H2O_AutoML' :
            test[self.target]=test[self.target].fillna(0).astype(float) ## Avoid H2O OS_Error
            train_h2o,test_h2o=h2o.H2OFrame(train),h2o.H2OFrame(test)
            if id_h2o_model == None:
                aml=H2OAutoML(max_models=self.hparameters['H2O_AutoML']['max_models'],
                              max_runtime_secs=self.hparameters['H2O_AutoML']['max_runtime_secs'],
                              seed=self.hparameters['H2O_AutoML']['seed'],
                              sort_metric=self.hparameters['H2O_AutoML']['sort_metric'],
                              exclude_algos=None)
                aml.train(x=list(test.columns)[:-1] ,y=self.target ,training_frame=train_h2o)
                self.id_h2o_leader=h2o.get_model(aml.leaderboard.as_data_frame()['model_id'][0])
                y_predict=self.id_h2o_leader.predict(test_h2o).asnumeric().as_data_frame()['predict']
                self.h2o_leaderboard = aml.leaderboard.as_data_frame()
            else: 
                y_predict=self.id_h2o_leader.predict(test_h2o).asnumeric().as_data_frame()['predict']
            
        elif algo == 'XGBoost':
            xg_params=self.hparameters['XGBoost']
            regressor_XG=xgboost.XGBRegressor(**xg_params)
            regressor_XG.fit(X_train, y_train)
            y_predict=regressor_XG.predict(X_test)
               
        return y_predict

    def univariate_forecast(self,
                            algo: str):
        """
        Generate univariate forecasts using specified algorithms.

        Args:
            algo (str): Algorithm to use for forecasting ('Prophet' or 'AutoArima').

        Returns:
            pd.DataFrame: Forecasted results.
        """
        forecasts,l_trues,l_preds=[],[],[]
        
        X=self.dataset.copy()
        
        X=super().slice_timestamp(X)
        
        if self.forecast_size <= 5 and 550>len(X)>=150: self.x_lag=5  
        if self.forecast_size <= 10 and 5000>len(X)>=550: self.x_lag=6  ## Review
        if self.forecast_size <= 5 and len(X)>=5000: self.x_lag=4
        if self.forecast_size >= 20: self.x_lag=2
        
        X=X.iloc[self.forecast_size*self.x_lag:,:]  ## Equal data lenght to Multivariate Approach given the rows drop in multivariate lags*  
        
        df_final=X[['Date', self.target]]
        df_final=df_final.rename(columns={'Date':'ds'})
        df_final[[self.target]]=df_final[[self.target]].astype(float)
        
        len_train=int((self.train_size*len(df_final)))
        len_test=int(len(df_final.iloc[len_train:,:]))
        
        train_set=df_final.iloc[:len_train,:]
        
        assert len_test>=self.forecast_size , 'len(Test)<=forecast_length, try to reduce your train_size ratio'
        
        iterations,iters=(int(len_test/self.sliding_size)),(int((len_test-self.forecast_size)/self.sliding_size))+1
        
        if algo=='AutoArima':
        
            aa_params=self.hparameters['AutoArima']
            model_arima=auto_arima(train_set[[self.target]], **aa_params)
        
        for rolling_cycle in range(0, iterations):
            
            if rolling_cycle==0:
                window=0
            else:
                window=self.sliding_size
            
            len_train=len_train+window
            
            train=df_final.iloc[:len_train,:]
            test=df_final.iloc[len_train:,:]
            
            if len(test[self.target])>=self.forecast_size:
                
                print('Algorithm Evaluation:', algo,'|| Window Iteration:', rolling_cycle+1, 'of', iters)
                
                print('Rows Train:', len(train))
                
                print('Rows Test:', len(test))
                
                train_=train.copy()
                
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
                
                if algo=='Prophet':
                    pr_params=self.hparameters['Prophet']
                    m=Prophet(**pr_params)
                    model_p=m.fit(train_)
                    
                    future=model_p.make_future_dataframe(periods=self.forecast_size,freq=frequency)
                    forecast=model_p.predict(future)
    
                    col='yhat'
                    y_pred=forecast.iloc[len(forecast)-self.forecast_size:,:]
    
                elif algo=='AutoArima':
                    
                    model_arima.fit(train_[[self.target]])
                
                    forecast=model_arima.predict(n_periods=self.forecast_size)
                    y_pred=pd.DataFrame(forecast, columns=[0])
                    col=0
                    
                y_true,y_pred=test[self.target][0:self.forecast_size],y_pred[col]    
                l_trues.append(y_true),l_preds.append(y_pred)
                x, y=pd.concat(l_trues).reset_index(drop=True), pd.concat(l_preds).reset_index(drop=True)
                
                forecasts.append(pd.DataFrame({'y_true':x, 
                                              'y_pred':y, 
                                              'Window':rolling_cycle+1})
                                .iloc[-self.forecast_size:]
                                .set_index(test['ds'].iloc[:self.forecast_size]))

        X_=pd.concat(forecasts)
    
        return X_
    
    
    def multivariate_forecast(self,            
                              algo: str):
        """
        Generate multivariate forecasts using specified algorithms.

        Args:
            algo (str): Algorithm to use for forecasting.

        Returns:
            pd.DataFrame: Forecasted results.
        """
        assert self.train_size>=0.3 and self.train_size<1 , 'train_size value should be in [0.3,1[ interval'
    
        forecasts,l_trues,l_preds=[],[],[]
        
        X=self.dataset.copy()
        
        X=super().slice_timestamp(X)
        
        if self.forecast_size <= 5 and 550>len(X)>=150: self.x_lag=5  
        if self.forecast_size <= 10 and 10000>len(X)>=550: self.x_lag=6  ## Review
        if self.forecast_size <= 5 and len(X)>=10000: self.x_lag=4
        if self.forecast_size >= 20: self.x_lag=2
        
        X_date=X.iloc[self.forecast_size*self.x_lag:,:]
        
        X=super().engin_date(X)
        
        X=super().multivariable_lag(X,
                                     range_lags=[self.forecast_size,self.forecast_size*self.x_lag],
                                     drop_na=True)
    
        df_final=X.copy()
        
        df_final[[self.target]]=df_final[[self.target]].astype(float)
        
        len_train=int((self.train_size*len(df_final)))
        len_test=int(len(df_final.iloc[len_train:,:]))
        
        assert len_test>=self.forecast_size , 'len(Test)<=forecast_length, try to reduce your train_size ratio'
        
        iterations,iters=(int((len_test)/self.sliding_size)),(int((len_test-self.forecast_size)/self.sliding_size))+1
          
        for rolling_cycle in range(0, iterations):
                
            if rolling_cycle==0: window=0
            else: window=self.sliding_size
            
            len_train=len_train+window
            
            train=df_final.iloc[:len_train,:]
            test,X_date_=df_final.iloc[len_train:,:],X_date.iloc[len_train:,:]
            
            if (len(test[self.target])>=self.forecast_size):
                
                print('Algorithm Evaluation:', algo,'|| Window Iteration:', rolling_cycle+1, 'of', iters)
                
                print('Rows Train:', len(train))
                
                print('Rows Test:', len(test))
                
                input_cols=list(train.columns)
                input_cols.remove(self.target)
    
                scaler=StandardScaler()
    
                scaler=scaler.fit(train[input_cols])
                train[input_cols]=scaler.transform(train[input_cols])
                test[input_cols]=scaler.transform(test[input_cols])            
                
                if algo=='H2O_AutoML' and rolling_cycle>=1:
                    y_pred=self.prediction(train=train,
                                           test=test,
                                           algo=algo,
                                           id_h2o_model=self.id_h2o_leader)
                else: y_pred=self.prediction(train=train,
                                             test=test,
                                             algo=algo)

                y_true,y_pred=pd.Series(test[self.target][0:self.forecast_size]),pd.Series(y_pred[0:self.forecast_size].tolist())
                
                l_trues.append(y_true),l_preds.append(y_pred)
                x,y=pd.concat(l_trues).reset_index(drop=True),pd.concat(l_preds).reset_index(drop=True)
                
                forecasts.append(pd.DataFrame({'y_true':x,
                                              'y_pred':y,
                                              'Window':rolling_cycle+1})
                                .iloc[-self.forecast_size:, :] 
                                .set_index(X_date_['Date'][0:self.forecast_size]))
    
        X_=pd.concat(forecasts)

        X_['y_pred']=pd.to_numeric(X_['y_pred'].astype(str).str.replace('[', '').str.replace(']', '')).round(5)
    
        return X_