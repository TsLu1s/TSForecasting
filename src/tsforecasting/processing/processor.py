import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import (
        RandomForestRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor
        )


class Processing:
    def __init__(self):
        self.target = 'y'
        self.date_cols = []
        self.input_cols = []
        self.target_cols = []
    
    def slice_timestamp(self, 
                        dataset:pd.DataFrame):
        """
        Detects all columns in the DataFrame that are date or datetime related, including those with timezones.
    
        Args:
            dataset (pd.DataFrame): Input DataFrame.
        """
        
        X = dataset.copy()

        # Loop through columns and detect different types of datetime data
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                self.date_cols.append(col)
                            
        for col in self.date_cols:
            # Format the datetime column
            X[col] = pd.to_datetime(X[col].dt.strftime('%Y-%m-%d %H:%M:%S'))
        
        if 'Date' in self.date_cols:
            X.index = X['Date']
                
        return X  
    
    def engin_date(self, 
                   dataset: pd.DataFrame,
                   drop: bool = True):
        """
        Engineer date-related features from datetime columns in the input DataFrame.
    
        Args:
            dataset (pd.DataFrame): Input DataFrame.
            drop (bool, optional): Whether to drop original datetime columns after feature creation. Defaults to True.
    
        Returns:
            pd.DataFrame: DataFrame with newly engineered date-related features.
        """

        # Identify columns that are datetime
        X = self.slice_timestamp(dataset).copy()
        
        # Dataframe to store new features
        X_ = pd.DataFrame(index=X.index)

        # Loop through each datetime column to create date-related features directly
        for col in list(set(self.date_cols)):

            X_[col + '_day_of_month'] = X[col].dt.day
            X_[col + '_day_of_week'] = X[col].dt.dayofweek + 1
            X_[col + '_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
            X_[col + '_month'] = X[col].dt.month
            X_[col + '_day_of_year'] = X[col].dt.dayofyear
            X_[col + '_year'] = X[col].dt.year
            X_[col + '_hour'] = X[col].dt.hour
            X_[col + '_minute'] = X[col].dt.minute
            X_[col + '_second'] = X[col].dt.second
            # Optionally drop the original datetime column
            if drop:
                X = X.drop(col, axis=1)
                
        # Concatenate the new features at the beginning of the DataFrame
        X = pd.concat([X_, X], axis=1)
    
        return X
    
    def make_timeseries(self, 
                    dataset: pd.DataFrame,
                    window_size: int, 
                    horizon: int, 
                    granularity: str = '1d',
                    datetime_engineering: bool = True):
        """
        Transforms a DataFrame into a single DataFrame containing sequential windows and horizon predictions,
        including NaNs where full horizon data isn't available. This allows using all available data for training up to the last point.
    
        Args:
            dataset (pd.DataFrame): A DataFrame with at least a 'y' column for time series data and a 'Date' column.
            window_size (int): The number of time steps in each window, indicating how many past observations each input sample includes.
            horizon (int): The number of time steps to predict into the future, which sets how far ahead the labels are relative to the end of each window.
            granularity (str): granularity of datetime column: 1m,30m,1h,1d,1wk,1mo (default='1d')
        Returns:
            pd.DataFrame: A DataFrame where each row represents a window of observations and horizon predictions,
                          starting with the date of the window, followed by lag features, and then horizon predictions.
    
        Raises:
            ValueError: If the `window_size` is larger than the input array length, which would make windowing impossible.
        """
        
        if window_size > len(dataset):
            raise ValueError("The lags length (window_size) cannot exceed the length of the time_series.")
    
        time_series = dataset[self.target].values
        date_series = dataset['Date'].values  
    
        # Calculate the maximum index for creating windows
        max_window_index = len(time_series) - window_size
    
        # Create windows and corresponding dates
        windows = np.array([time_series[i:i + window_size] for i in range(max_window_index + 1)])
        window_dates = date_series[window_size - 1:]  # Date corresponding to the end of each window
    
        # Create horizons, allowing NaNs for the future windows
        horizons = np.array([time_series[i + window_size:i + window_size + horizon] if i + window_size + horizon <= len(time_series)
                             else np.concatenate([time_series[i + window_size:], np.full((i + window_size + horizon - len(time_series)), np.nan)])
                             for i in range(max_window_index + 1)])
    
        # Add future NaN horizons
        future_dates = [pd.to_datetime(date_series[-1])] #+ pd.Timedelta(granularity) * (i + 1) for i in range(horizon)]
        future_windows = [time_series[-window_size:] for _ in range(horizon)]
        future_horizons = np.array([np.full((horizon,), np.nan) for _ in range(horizon)])
    
        # Combine past and future windows and horizons
        windows = np.concatenate([windows, future_windows[:1]])
        horizons = np.concatenate([horizons, future_horizons[:1]])
        window_dates = np.concatenate([window_dates, future_dates[:1]])
    
        # Ensure window_dates are datetime
        window_dates = pd.to_datetime(window_dates)
    
        # Column names for windows and horizons
        self.lag_cols = [f'y_lag_{i+1}' for i in range(window_size)]
        self.horizon_cols = [f'y_horizon_{i+1}' for i in range(horizon)]
    
        # Create DataFrame for lags and horizons
        lag_df = pd.DataFrame(windows, columns=self.lag_cols)
        horizon_df = pd.DataFrame(horizons, columns=self.horizon_cols)
    
        # Combine into one DataFrame
        X = pd.concat([lag_df, horizon_df], axis=1).reset_index(drop=True)
        X.insert(0, 'Date', window_dates)
        
        if datetime_engineering:
            X = self.engin_date(dataset = X,
                                drop = True)
        X = X.head(X.shape[0]-1)
        
        self.target_cols = [col for col in X.columns if 'horizon' in col]
        self.input_cols = [col for col in X.columns if col not in self.target_cols]
        
        return X
    
    def future_timestamps(self, dataset: pd.DataFrame, horizon: int, granularity: str = '1d'):
        """
        Generates future timestamps based on the last timestamp in the input dataset.
    
        Args:
            dataset (pd.DataFrame): Input DataFrame with a 'Date' column.
            horizon (int): Number of future timestamps to generate.
            granularity (str, optional): Granularity of timestamps ('1m', '30m', '1h', '1d', '1wk', '1mo'). Defaults to '1d'.
    
        Returns:
            pd.DataFrame: DataFrame with generated future timestamps.
        """
        
        X = self.slice_timestamp(dataset).copy()
        
        last_date = pd.to_datetime(X['Date'].iloc[-1])
        timestamp_list = []
    
        if granularity == '1m':
            delta = datetime.timedelta(minutes=1)
        elif granularity == '30m':
            delta = datetime.timedelta(minutes=30)
        elif granularity == '1h':
            delta = datetime.timedelta(hours=1)
        elif granularity == '1d':
            delta = datetime.timedelta(days=1)
        elif granularity == '1wk':
            delta = datetime.timedelta(weeks=1)
        elif granularity == '1mo':
            delta = relativedelta(months=1)
        else:
            raise ValueError("Unsupported granularity provided.")
    
        for _ in range(horizon):
            last_date += delta
            timestamp_list.append(last_date)
    
        return pd.DataFrame({'Date': timestamp_list})

    @staticmethod
    def treebased_feature_selection(dataset: pd.DataFrame,
                                    target: str = 'y',
                                    relevance: float = 0.99,
                                    algo: str = 'ExtraTrees',
                                    estimators: int = 250):
        """
        Conducts feature selection based on the importance derived from specified tree-based regression models.
        This method aims to retain a subset of features whose cumulative importance meets a predefined threshold, thus facilitating model interpretability and efficiency.
    
        Parameters:
            dataset (pd.DataFrame): The dataset containing both features and the target variable.
            target (str, optional): The name of the target variable column. Defaults to 'y'.
            relevance (float, optional): The cumulative feature importance threshold to retain. Valid values range from 0.5 to 1.0. Defaults to 0.99.
            algo (str, optional): Specifies the tree-based regression algorithm to use for assessing feature importance. 
                                  Options include 'ExtraTrees', 'RandomForest', and 'GBR' (Gradient Boosting Regressor). Defaults to 'ExtraTrees'.
            estimators (int, optional): The number of trees to build in the model. More trees can increase the accuracy but also the computational cost. Defaults to 250.
    
        Returns:
            tuple:
                - list: The names of the selected columns that meet the relevance threshold.
                - pd.DataFrame: A DataFrame with two columns: 'variable' and 'percentage', indicating the feature names and their respective importances.
    
        Raises:
            AssertionError: If 'relevance' is not between 0.5 and 1.0.
        """
        # Validate the relevance input to ensure it is within the acceptable range.
        assert 0.5 <= relevance <= 1, 'Relevance value should be within the [0.5, 1.0] interval.'
    
        # Preparing the data, separating features and target
        train = dataset.copy()
        features = [col for col in train.columns if col != target]
        X_train = train[features].values
        y_train = train[target].values
        
        # Selecting the model based on the algorithm specified
        if algo == 'RandomForest':
            model = RandomForestRegressor(n_estimators=estimators)
        elif algo == 'ExtraTrees':
            model = ExtraTreesRegressor(n_estimators=estimators)
        elif algo == 'GBR':
            model = GradientBoostingRegressor(n_estimators=estimators)
        else:
            raise ValueError(f"Unsupported algorithm '{algo}'. Choose from 'RandomForest', 'ExtraTrees', 'GBR'.")
    
        # Fitting the model and obtaining feature importances
        model.fit(X_train, y_train)
        importances = model.feature_importances_
    
        # Creating a DataFrame of feature importances
        feat_imp_df = pd.DataFrame({
            'variable': features,
            'percentage': importances
        }).sort_values(by='percentage', ascending=False)
        
        # Determining the features whose cumulative importance meets the relevance threshold
        cumulative_importance = 0.0
        selected_features = []
        for _, row in feat_imp_df.iterrows():
            selected_features.append(row['variable'])
            cumulative_importance += row['percentage']
            if cumulative_importance >= relevance:
                break
    
        # Return the list of selected feature names and the feature importance DataFrame
        return selected_features, feat_imp_df







