import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsforecasting.processing.processor import Processing 
from tsforecasting.models.forecasting_models import (RandomForest_Forecasting, 
                                                     ExtraTrees_Forecasting,
                                                     GBR_Forecasting,
                                                     KNN_Forecasting,
                                                     GeneralizedLR_Forecasting,
                                                     XGBoost_Forecasting,
                                                     CatBoost_Forecasting,
                                                     LightGBM_Forecasting,
                                                     AutoGluon_Forecasting)

class Training: 
    """
    This class is designed to encapsulate the training process for time series forecasting models.
    It supports data preparation, model training, prediction, and evaluation using a variety of
    machine learning models. The class allows for flexible specification of model hyperparameters,
    dataset features, training/test splits, and forecasting horizons.
    """
    def __init__(self,
                 dataset: pd.DataFrame,
                 train_size: float,
                 lags: int, 
                 horizon: int,
                 sliding_size: int,
                 hparameters: dict = {},
                 granularity: str = '1d'):
        """
        Initializes the Training class with the required dataset and parameters for model training.

        Parameters:
        - dataset: DataFrame containing the input data.
        - train_size: Fraction of the dataset to use for training the model.
        - lags: Number of previous time steps to use as input features.
        - horizon: Number of time steps to predict into the future.
        - sliding_size: The window size for each step in a rolling forecast scenario.
        - hparameters: Dictionary of model-specific hyperparameters.
        - granularity: Time granularity of data, defaulting to daily ('1d').
        """

        self.dataset = dataset
        self.train_size = train_size
        self.lags = lags
        self.horizon = horizon
        self.sliding_size = sliding_size
        self.hparameters = hparameters
        self.granularity = granularity
        self.target = "y" # Default target variable name, can be customized if needed.
        self.input_cols = []
        self.target_cols = []
        self.forecasts = [] # List to accumulate forecast results.
        self.autogluon_fit = None # Flag to check if AutoGluon model is fitted.
        self.regressor_AG = None # Placeholder for an AutoGluon regressor instance.
        self.processing = Processing() # Initialize an instance of the Processing class.
        
    def fit_predict(self,
                    train: pd.DataFrame,
                    test: pd.DataFrame,
                    algo: str = 'RandomForest'):
        """
        Fits a specified forecasting model to the training data and predicts outcomes for the test data.

        Parameters:
        - train: DataFrame containing the training data.
        - test: DataFrame containing the testing data.
        - algo: String identifier for the model to use.

        Returns:
        - y_predict: DataFrame containing the predicted values.
        """
        
        # Extract input features and targets from training and test data.
        X_train = train[self.input_cols].values
        X_test = test[self.input_cols].values
        y_train = train[self.target_cols].values
        
        # Select and execute the modeling process based on the 'algo' parameter.
        if algo == 'RandomForest' :
            rf_params = self.hparameters['RandomForest']
            regressor_RF = RandomForest_Forecasting(**rf_params)
            regressor_RF.fit(X_train, y_train)
            y_predict = regressor_RF.predict(X_test)
            
        elif algo == 'ExtraTrees' :
            et_params = self.hparameters['ExtraTrees']
            regressor_ET = ExtraTrees_Forecasting(**et_params)
            regressor_ET.fit(X_train, y_train)
            y_predict = regressor_ET.predict(X_test)
            
        elif algo == 'GBR' :
            gbr_params = self.hparameters['GBR']
            regressor_GBR = GBR_Forecasting(**gbr_params)
            regressor_GBR.fit(X_train, y_train)
            y_predict = regressor_GBR.predict(X_test)
            
        elif algo == 'KNN' :
            knn_params = self.hparameters['KNN']
            regressor_KNN = KNN_Forecasting(**knn_params)
            regressor_KNN.fit(X_train, y_train)
            y_predict = regressor_KNN.predict(X_test)
            
        elif algo == 'GeneralizedLR' :
            glr_params = self.hparameters['GeneralizedLR']
            regressor_TD = GeneralizedLR_Forecasting(**glr_params)
            regressor_TD.fit(X_train, y_train)
            y_predict = regressor_TD.predict(X_test)
            
        elif algo == 'XGBoost':
            xg_params = self.hparameters['XGBoost']
            regressor_XG = XGBoost_Forecasting(**xg_params)
            regressor_XG.fit(X_train, y_train)
            y_predict = regressor_XG.predict(X_test)
        
        elif algo == 'Catboost':
            cb_params = self.hparameters['Catboost']
            regressor_XG = CatBoost_Forecasting(**cb_params)
            regressor_XG.fit(X_train, y_train)
            y_predict = regressor_XG.predict(X_test)
        
        elif algo == 'LightGBM':
            lgbm_params = self.hparameters['Lightgbm']
            regressor_XG = LightGBM_Forecasting(**lgbm_params)
            regressor_XG.fit(X_train, y_train)
            y_predict = regressor_XG.predict(X_test)
        
        elif algo == 'AutoGluon':
            if not self.autogluon_fit:                
                ag_params = self.hparameters['AutoGluon']
                print(" ")
                self.regressor_AG = AutoGluon_Forecasting(labels=self.target_cols, 
                                                          eval_metric=ag_params.get('eval_metric'), 
                                                          verbosity=ag_params.get('verbosity'),
                                                          presets=ag_params.get('presets'))
                self.regressor_AG.fit(train_data = train,
                                      time_limit = ag_params.get('time_limit'), 
                                      save_space = ag_params.get('save_space'),
                                      )  
                self.autogluon_fit = True
                y_predict = self.regressor_AG.predict(test).astype(float)
            else:
                y_predict = self.regressor_AG.predict(test).astype(float)
            y_predict.columns = ['y_forecast_' + name.split('_')[1] + '_' + name.split('_')[2] 
                                 if name.startswith('y_horizon') else name for name in y_predict.columns]
        if algo != 'AutoGluon':
            # Create column names based on the number of columns
            forecast_cols = [f'{self.target}_forecast_horizon_{i+1}' for i in range(y_predict.shape[1])]
            
            # Create the DataFrame with these column names
            y_predict = pd.DataFrame(y_predict, columns=forecast_cols, index=test.index)
        
        return y_predict
    
    def multivariate_forecast(self,            
                              algo: str):
        """
        Conducts a rolling forecast using the specified model, allowing for evaluation over multiple
        iterations and dynamic adjustment of the training window.

        Parameters:
        - algo: String identifier for the model to use.

        Ensures the conditions for a valid multivariate forecast are met, prepares the dataset, and
        handles rolling forecasts with specified settings. Each forecast is accumulated and finally
        combined into a comprehensive DataFrame returned as the output.
        """
        # Ensure preconditions for training and forecast are met.
        assert self.train_size >= 0.3 and self.train_size < 1 , 'train_size value should be in [0.3,1[ interval'
        assert self.dataset.shape[0] >= self.horizon + self.lags , 'Length of test is incompatible with lags & horizon values'
        
        # Prepare the dataset by resetting the index and creating time series features.
        X = self.dataset.reset_index(drop=True).copy()
        
        X = self.processing.make_timeseries(dataset=X, window_size=self.lags, horizon=self.horizon, granularity=self.granularity)
        
        # Identify and separate input and target columns based on the processed dataset.
        self.target_cols = [col for col in X.columns if 'horizon' in col]
        self.input_cols = [col for col in X.columns if col not in self.target_cols]  
                
        # Calculate the number of data points for training based on train_size.
        df = X[:-self.horizon].copy()
        train_size = int(self.train_size*df.shape[0])
        iterations = int(df.iloc[train_size:,:].shape[0]/self.sliding_size)
         
        # Conduct rolling window forecasting.
        for rolling_cycle in range(0, iterations):
            # Calculate the starting index for each new training set incrementally.
            if rolling_cycle == 0:
                window = 0
            else: 
                window = self.sliding_size
            
            train_size = train_size + window
            
            train = df.iloc[:train_size,:]
            test = df.iloc[train_size:,:]
            
            if test.shape[0]>=self.horizon:
                
                print('Algorithm Evaluation:', algo,'|| Window Iteration:', rolling_cycle+1, 'of', iterations)
                print('Rows Train:', train.shape[0])
                print('Rows Test:', test.shape[0])
                
                if rolling_cycle+1 == iterations:
                    print(" ")
                    
                # Scale features using StandardScaler before training.
                scaler = StandardScaler()
                scaler = scaler.fit(train[self.input_cols])
                train[self.input_cols] = scaler.transform(train[self.input_cols])
                test[self.input_cols] = scaler.transform(test[self.input_cols])
                
                # Call fit_predict to train the model and make forecasts.
                y_pred = self.fit_predict(train=train,
                                            test=test,
                                            algo=algo)
                y_trues, y_preds = test[self.target_cols][:self.horizon], y_pred[:self.horizon]
                # Append forecasts to the list for aggregation.
                self.forecasts.append(pd.concat([y_trues, y_preds], axis=1).assign(Window=rolling_cycle + 1, Model=algo))
    
        return pd.concat(self.forecasts)
    
