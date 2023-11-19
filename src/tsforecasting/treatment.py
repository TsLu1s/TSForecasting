import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
    )


class Treatment:
    def __init__(self):
        self.target = 'y'

    def round_cols(self,
                   dataset: pd.DataFrame,
                   round_: int = 5):
    
        X=dataset.copy()
        
        num_cols=[col for col in X.select_dtypes(include=['float', 'float64','float32']).columns if col != self.target]
        for col in num_cols: X[[col]]=X[[col]].round(round_)
            
        return X
    
    def slice_timestamp(self, 
                        dataset: pd.DataFrame):
        """
        Extracts timestamp column 'Date' and moves target column to the end of the DataFrame.
        Converts timestamp column into specific format ('%Y-%m-%d %H:%M:%S').
        Args:
            dataset (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with datetime timestamp formatted and target column moved to the end.
        """
        X=dataset.copy()
        X=X[[col for col in X.columns if col != self.target] + [self.target]]
        
        for col in list(X.columns):
            if col=='Date': X['Date']=pd.to_datetime(X['Date'].dt.strftime('%Y-%m-%d %H:%M:%S'))
        return X
    

    def multivariable_lag(self,
                          dataset: pd.DataFrame,
                          range_lags: list = [1,10],
                          drop_na: bool = True):
        """
        Creates lag features for the target "y" variable within a specified range of lags.

        Args:
            dataset (pd.DataFrame): Input DataFrame.
            range_lags (list, optional): Range of lag intervals. Defaults to [1, 10].
            drop_na (bool, optional): Whether to drop rows with NA values. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with lag features added.
        """
        assert range_lags[0]>=1, 'Range lags first interval value should be bigger then 1'
        assert range_lags[0]<=range_lags[1], 'Range lags first interval value should be bigger then second'
        
        X=dataset.copy()
        X_=X.copy()
        X_=self.slice_timestamp(X_)
    
        for e in range(range_lags[0],range_lags[1]+1):
            lag_str=str(e)
            lag_str=self.target+'_lag_'+lag_str
            df=pd.DataFrame({'target':X[self.target],
                          lag_str:X[self.target].shift(e)
                          })
            df=df.drop('target',axis=1)
            X_[lag_str]=df[lag_str]
        cols,input_cols=list(X_.columns),list(X_.columns)
        input_cols.remove(self.target)
        last_col=cols[len(cols)-1:]
        if drop_na==True:
            X_=X_.dropna(axis=0, subset=last_col)
        elif drop_na==False:
            X_[input_cols]=X_[input_cols].apply(lambda x: x.fillna(x.mean()),axis=0)
        for col in cols:
            if col=='Date':
                X_=X_.set_index(['Date']) 
                break
            
        return X_
    
    def future_timestamps(self,
                          dataset: pd.DataFrame,
                          forecast_size: int,
                          granularity: str ='1d'):
        """
        Generates future timestamps based on the last timestamp in the input dataset.
        
        Args:
            dataset (pd.DataFrame): Input DataFrame with a 'Date' column.
            forecast_size (int): Number of future timestamps to generate.
            granularity (str, optional): Granularity of timestamps ('1m', '30m', '1h', '1d', '1wk', '1mo'). Defaults to '1d'.
        
        Returns:
            pd.DataFrame: DataFrame with generated future timestamps.
        """
        X=dataset.copy()
        X=self.slice_timestamp(X)

        timestamp,timestamp_list=list((X.Date[len(X)-1:]))[0],[]
        
        if granularity=='1m':
        
            def generate_datetimes(date_from_str=timestamp, days=1000):
               date_from=datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
               for n in range(1,1420*days):
                   yield date_from + datetime.timedelta(minutes=n)
            for date in generate_datetimes():
                timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))
                
        elif granularity=='30m':
        
            def generate_datetimes(date_from_str=timestamp, days=1000):
               date_from=datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
               for n in range(1,1420*days):
                   yield date_from + datetime.timedelta(minutes=30*n)
            for date in generate_datetimes():
                timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))
        
        elif granularity=='1h':
        
            def generate_datetimes(date_from_str=timestamp, days=1000):
               date_from=datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
               for hour in range(1,24*days):
                   yield date_from + datetime.timedelta(hours=hour)
            for date in generate_datetimes():
                timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))
            
        elif granularity=='1d':
        
            def generate_datetimes(date_from_str=timestamp, days=1000):
               date_from=datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
               for day in range(1,days):
                   yield date_from + datetime.timedelta(days=day)
            for date in generate_datetimes():
                timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))

        elif granularity=='1wk':
                
            def generate_datetimes(date_from_str=timestamp, weeks=1000):
               date_from=datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
               for week in range(1,weeks):
                   yield date_from + datetime.timedelta(weeks=week)
            for date in generate_datetimes():
                timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))
        
        elif granularity=='1mo':
        
            def generate_datetimes(date_from_str=timestamp, months=100):
               date_from=datetime.datetime.strptime(str(date_from_str), '%Y-%m-%d %H:%M:%S')
               for month in range(0,months):
                   yield date_from + relativedelta(months=month+1)
            for date in generate_datetimes():
                timestamp_list.append((date.strftime('%Y-%m-%d %H:%M:%S')))
        
        X_=pd.DataFrame()
        X_['Date']=timestamp_list
        X_['Date']=pd.to_datetime(X_['Date'])
        
        X_=X_.iloc[0:forecast_size,:]
        
        return X_

    @staticmethod
    def engin_date(dataset: pd.DataFrame,
                   drop: bool = True):
        """
        Engineer date-related features in the input DataFrame.

        Args:
            dataset (pd.DataFrame): Input DataFrame.
            drop (bool, optional): Whether to drop original datetime columns. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with engineered date-related features.
        """
        # This method is responsible for engineering date-related features in the input DataFrame X.
        # It can also optionally drop the original datetime columns based on the 'drop' parameter.
        X=dataset.copy()
        # Extract the data types of each column in X and create a DataFrame x.
        x=pd.DataFrame(X.dtypes)
        # Create a 'column' column to store the column names.
        x['column']=x.index
        # Reset the index and drop the original index column.
        x=x.reset_index().drop(['index'], axis=1).rename(columns={0:'dtype'})
        # Filter for columns with datetime data type.
        a=x.loc[x['dtype'] == 'datetime64[ns]']
    
        # Initialize an empty list to store the names of datetime columns.
        date_columns=[]
    
        # Loop through datetime columns.
        for date_col in a['column']:
            date_columns.append(date_col)
            # Convert datetime values to a standardized format (Year-Month-Day Hour:Minute:Second).
            X[date_col]=pd.to_datetime(X[date_col].dt.strftime('%Y-%m-%d %H:%M:%S'))
    
        # Define a function to create additional date-related features for a given column.
        def create_date_features(X, col):
            # Extract day of the month, day of the week, and whether it's a weekend.
            X[col + '_day_of_month']=X[col].dt.day
            X[col + '_day_of_week']=X[col].dt.dayofweek + 1
            X[[col + '_is_wknd']]=X[[col + '_day_of_week']].replace([1, 2, 3, 4, 5, 6, 7],
                                                                                  [0, 0, 0, 0, 0, 1, 1])
            X[col + '_month']=X[col].dt.month
            X[col + '_day_of_year']=X[col].dt.dayofyear
            X[col + '_year']=X[col].dt.year
            X[col + '_hour']=X[col].dt.hour
            X[col + '_minute']=X[col].dt.minute
            X[col + '_second']=X[col].dt.second
    
            return X
    
        # Loop through the list of date columns and create the additional features using the defined function.
        for col in date_columns:
            X=create_date_features(X, col)
            # If 'drop' is set to True, drop the original datetime column.
            if drop == True: X=X.drop(col, axis=1)
    
        # Return the DataFrame X with the engineered date-related features.
        return X
    

    @staticmethod
    def feature_selection_tb(dataset: pd.DataFrame,
                             target: str = 'y',
                             relevance: float = 0.99,
                             algo: str = 'ExtraTrees',
                             estimators: int = 250):
        """
        Perform feature selection using tree-based algorithms (RandomForest,ExtraTrees,GBR).

        Args:
            dataset (pd.DataFrame): Input DataFrame.
            target (str, optional): Target variable column name. Defaults to 'y'.
            relevance (float, optional): Total feature importance to retain. Defaults to 0.99.
            algo (str, optional): Tree-based algorithm ('ExtraTrees', 'RandomForest', 'GBR'). Defaults to 'ExtraTrees'.
            estimators (int, optional): Number of estimators for the algorithm. Defaults to 250.

        Returns:
            tuple: Selected columns and DataFrame with feature importances.
        """
        assert relevance>=0.5 and relevance<=1 , 'relevance value should be in [0.5,1[ interval'
        
        train=dataset.copy()
        train=train[[col for col in train.columns if col != target] + [target]] # target to last index
       
        X_train=train.iloc[:, 0:(len(list(train.columns))-1)].values
        y_train=train.iloc[:, (len(list(train.columns))-1)].values
        
        if algo=='RandomForest':
            fs_model=RandomForestRegressor(n_estimators=estimators)
            fs_model.fit(X_train, y_train)
        elif algo=='ExtraTrees':
            fs_model=ExtraTreesRegressor(n_estimators=estimators)
            fs_model.fit(X_train, y_train)
        elif algo=='GBR':
            fs_model=GradientBoostingRegressor(n_estimators=estimators)
            fs_model.fit(X_train, y_train)
    
        column_imp=fs_model.feature_importances_
        column_names=list(train.columns).copy()
        column_names.remove(target)
        
        Columns,feat_imp=pd.Series(column_names),pd.Series(column_imp)
        
        X=pd.concat([feat_imp,Columns],axis=1)
        X=X.rename(columns={0:'percentage',1:'variable'})
        
        n=0.015
        va_df=X[X['percentage'] > n]
        val=va_df['percentage'].sum()
        for iteration in range(0,10):
                
            if val<=relevance:
                va_df=X[X['percentage']>n]
                n=n*0.5
                val=va_df['percentage'].sum()
            elif val>relevance:
                break
        va_df=va_df.sort_values(['percentage'], ascending=False)
    
        sel_cols=[]
        for rows in va_df['variable']: sel_cols.append(rows)
        sel_cols.append(target)
        
        return sel_cols, va_df





