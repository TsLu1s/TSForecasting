import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsforecasting.configs.parameters import model_configurations 
from tsforecasting.processing.processor import Processing 
from tsforecasting.training.validation import Training 
from tsforecasting.performance.evaluation import vertical_performance, best_model 

class TSForecasting:
    """
    TSForecasting class integrates various functionalities required for effective time series forecasting.
    It includes setting up the model environment, handling data preprocessing, model training, evaluation,
    and making future forecats. The class allows using multiple machine learning algorithms and performs
    comparative analysis to select the best performing model based on specified evaluation growing forecasting window metrics.
    """
    def __init__(self,
                 train_size: float = 0.8,
                 lags: int = 7,
                 horizon: int = 15,
                 sliding_size: int = 15,
                 models: list = ['RandomForest','GBR','XGBoost'],
                 hparameters: dict = model_configurations(),
                 granularity: str = '1d',
                 metric: str = 'MAE'):
        """
        Initializes the forecasting framework with customizable settings.

        Args:
        train_size (float): The proportion of the dataset to be used for training the models.
        lags (int): The number of past time steps to use as features for forecasting.
        horizon (int): The number of future time steps to predict.
        sliding_size (int): The size of the step to move forward in the time series for the rolling window approach.
        models (list): List of model names to be used for forecasting.
        hparameters (dict): A dictionary containing hyperparameters for each model.
        granularity (str): The granularity of the time series data, e.g., '1d' for daily.
        metric (str): The metric to use for evaluating model performance.
        """
        self.train_size = train_size
        self.lags = lags
        self.horizon = horizon
        self.sliding_size = sliding_size
        self.hparameters = hparameters
        self.granularity = granularity
        self.models = models
        self.metric = metric
        self.timeseries = None
        self.target = 'y'
        self.list_models = ['RandomForest','ExtraTrees','GBR','KNN','GeneralizedLR',
                            'XGBoost', 'LightGBM','Catboost', 'AutoGluon']
        self.fit_performance = None
        self.fit_predictions = None
        self.selected_model = None
        self.scaler = StandardScaler() # Initialize the scaler for data normalization.
        self.processing = Processing() # Initialize data processing utility.
        self.validation = Training(dataset = pd.DataFrame(),  # Initialize the Training class with the specified settings.
                                   train_size = self.train_size,
                                   lags = self.lags,
                                   horizon = self.horizon,
                                   sliding_size = self.sliding_size,
                                   hparameters = self.hparameters,
                                   granularity = self.granularity)
        
    def fit_forecast(self,
                     dataset:pd.DataFrame):
        """
        Fits the specified models to the dataset and performs forecasting. This method processes the dataset,
        trains each specified model, evaluates them, and selects the best model based on the predefined metric.

        Args:
        dataset (pd.DataFrame): The time series dataset used for training and forecasting.

        Returns:
        self (TSForecasting): An instance of TSForecasting with updated attributes after fitting models.
        """
        self.validation.dataset, self.dataset = dataset.copy(), dataset.copy()
        forecasts = []
        for model in self.models: # Iterate over each model specified in the initialization.
            if model in self.list_models: # Check if the model is supported. 
                results = self.validation.multivariate_forecast(algo=model) # Run forecasting.
                forecasts.append(results) # Collect forecasting results.
                
        self.fit_predictions = forecasts[-1]
        self.complete_fit_performance = vertical_performance(self.fit_predictions,self.horizon).reset_index(drop=True)
        
        # Evaluate and select the best model based on the forecast results and the specified metric.
        self.selected_model = best_model(self.complete_fit_performance,metric=self.metric)
        
        return self # Return the class instance itself for potential method chaining.
    
    def history(self):
        """
        Compiles and returns the historical performance data for all fitted models. This method aggregates
        performance data across different forecasting horizons and models, providing a detailed view of model
        effectiveness over time.

        Returns:
        dict: A dictionary containing detailed performance data and model rankings.
        """
        # Dictionary to map short metric names to detailed descriptions.
        metric_mapping = { 
            'MAE': 'Mean Absolute Error',
            'MAPE': 'Mean Absolute Percentage Error',
            'MSE': 'Mean Squared Error'
        }
        _metric = metric_mapping.get(self.metric)       
        # Compute aggregated performance metrics across all models and horizons.
        self.aggregated_fit_performance = self.complete_fit_performance.groupby(['Model', 'Horizon']).agg({
                                                                                                          'Mean Absolute Error': 'mean',    # Mean for MAE
                                                                                                          'Mean Absolute Percentage Error': 'mean',    # Mean for MAPE
                                                                                                          'Mean Squared Error': 'mean',   # Mean for RMSE
                                                                                                          'Max Error': 'max'}).reset_index() # Max for Max Error
        # Create a leaderboard of models based on their performance.
        self.leaderboard = self.complete_fit_performance.groupby(['Model']).mean().reset_index() \
                                                                           .drop(['Window','Horizon','Max Error'],axis=1) \
                                                                           .sort_values(by=_metric, ascending=True)
        
        self.performance = {'Predictions': self.fit_predictions,
                            'Performance Complete': self.complete_fit_performance,
                            'Performance by Horizon': self.aggregated_fit_performance,
                            'Leaderboard': self.leaderboard,
                            }
        
        return self.performance
    
    def forecast(self):
        """
        Uses the selected best model to make future predictions. This method also computes confidence intervals
        for the forecasts based on historical model performance.

        Returns:
        pd.DataFrame: A DataFrame containing the forecasted values along with upper and lower confidence intervals.
        """
        X = self.dataset.copy()
        # Prepare future timestamps based on the specified horizon and granularity.
        forecast = self.processing.future_timestamps(X,
                                                     self.horizon,
                                                     self.granularity)
        
        me1, me2 = 'Mean Absolute Error','Max Error'
                
        X = self.processing.make_timeseries(dataset = X, window_size = self.lags, horizon = self.horizon)
        
        self.timeseries = X.copy() # Save the feature-engineered time series for potential future use.
        
        train, test = X.iloc[:-self.horizon,:], X.tail(1) # Split the data into fitting and future forecast sets.
        
        # Standardize features using StandardScaler
        self.scaler = self.scaler.fit(train[self.processing.input_cols])
        
        train[self.processing.input_cols] = self.scaler.transform(train[self.processing.input_cols])
        test[self.processing.input_cols] = self.scaler.transform(test[self.processing.input_cols])
        
        # Perform forecasting using the selected model and integrate forecasted values.
        forecast['y'] = self.validation.fit_predict(train = train,
                                                    test = test,
                                                    algo = self.selected_model).transpose().reset_index(drop=True) 
        
        intervals = self.performance['Performance by Horizon'][self.performance['Performance by Horizon']['Model'] == self.selected_model].reset_index(drop=True)
        # Compute confidence intervals for the forecast based on model performance.
        forecast['y_superior'], forecast['y_inferior'] = forecast['y'].add(intervals[me1]), forecast['y'].sub(intervals[me1])
        forecast['y_max_interval'], forecast['y_min_interval'] = forecast['y'].add(intervals[me2]), forecast['y'].sub(intervals[me2])
        
        return forecast


__all__ = [
    'Processing',
    'Training',
    'vertical_performance',
    'best_model',
    'model_configurations'
]