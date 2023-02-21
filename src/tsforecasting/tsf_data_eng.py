import pandas as pd
import numpy as np
import sys
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

def slice_timestamp(Dataset:pd.DataFrame,date_col:str='Date'):
    """
    The slice_timestamp function takes a dataframe and returns the same dataframe with the date column sliced to just include
    the year, month, day and hour. This is done by converting all of the values in that column to strings then slicing them 
    accordingly. The function then converts those slices back into datetime objects so they can be used for further analysis.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be sliced
    :param date_col:str='Date': Specify the name of the column that contains the date information
    :return: A dataframe with the timestamp column sliced to only include the year, month and day
    """
    df=Dataset.copy()
    cols=list(df.columns)
    for col in cols:
        if col==date_col:
            df[date_col] = df[date_col].astype(str)
            df[date_col] = df[date_col].str.slice(0,19)
            df[date_col] = pd.to_datetime(df[date_col])
    return df

def round_cols(Dataset:pd.DataFrame,
               target,round_:int=4):
    """
    The round_cols function rounds the numeric columns of a Dataset to a specified number of decimal places.
    The function takes two arguments:
    Dataset: A pandas DataFrame object.
    target: The name of the target column in the dataframe that will not be rounded.  This argument is required for safety reasons, so that no other columns are accidentally rounded.
    
    :param Dataset:pd.DataFrame: Pass the dataframe that will be transformed
    :param target: Indicate the target variable
    :param round_:int=4: Round the numbers to a certain number of decimal places
    :return: A dataframe with the same columns as the original one, but with all numeric columns rounded to 4 decimals
    """
    df_=Dataset.copy()
    df_round=df_.copy()
    list_num_cols=df_round.select_dtypes(include=['float']).columns.tolist()
    
    for elemento in list_num_cols:
        if elemento==target:
            list_num_cols.remove(target)
    for col in list_num_cols:
        df_round[[col]]=df_round[[col]].round(round_)
        
    return df_round

def engin_date(Dataset:pd.DataFrame,
               Drop:bool=False):
    """
    The engin_date function takes a DataFrame and returns a DataFrame with the date features engineered.
    The function has two parameters: 
    Dataset: A Pandas DataFrame containing at least one column of datetime data. 
    Drop: A Boolean value indicating whether or not to drop the original datetime columns from the returned dataset.
    
    :param Dataset:pd.DataFrame: Pass the dataset
    :param Drop:bool=False: Drop the original timestamp columns
    :return: The dataframe with the date features generated
    """
    Dataset_=Dataset.copy()
    Df=Dataset_.copy()
    Df=slice_timestamp(Df)
    
    x=pd.DataFrame(Df.dtypes)
    x['column'] = x.index
    x=x.reset_index().drop(['index'], axis=1).rename(columns={0: 'dtype'})
    a=x.loc[x['dtype'] == 'datetime64[ns]']

    list_date_columns=[]
    for col in a['column']:
        list_date_columns.append(col)

    def create_date_features(df,elemento):
        
        df[elemento + '_day_of_month'] = df[elemento].dt.day
        df[elemento + '_day_of_week'] = df[elemento].dt.dayofweek + 1
        df[[elemento + '_is_wknd']] = df[[elemento + '_day_of_week']].replace([1, 2, 3, 4, 5, 6, 7], 
                            [0, 0, 0, 0, 0, 1, 1 ]) 
        df[elemento + '_month'] = df[elemento].dt.month
        df[elemento + '_day_of_year'] = df[elemento].dt.dayofyear
        df[elemento + '_year'] = df[elemento].dt.year
        df[elemento + '_hour']=df[elemento].dt.hour
        df[elemento + '_minute']=df[elemento].dt.minute
        df[elemento + '_Season']=''
        winter = list(range(1,80)) + list(range(355,370))
        spring = range(80, 172)
        summer = range(172, 264)
        fall = range(264, 355)

        df.loc[(df[elemento + '_day_of_year'].isin(spring)), elemento + '_Season'] = '2'
        df.loc[(df[elemento + '_day_of_year'].isin(summer)), elemento + '_Season'] = '3'
        df.loc[(df[elemento + '_day_of_year'].isin(fall)), elemento + '_Season'] = '4'
        df.loc[(df[elemento + '_day_of_year'].isin(winter)), elemento + '_Season'] = '1'
        df[elemento + '_Season']=df[elemento + '_Season'].astype(np.int64)
        
        return df 
    
    if Drop==True:
        for elemento in list_date_columns:
            Df=create_date_features(Df,elemento)
            Df=Df.drop(elemento,axis=1)
    elif Drop==False:
        for elemento in list_date_columns:
            Df=create_date_features(Df,elemento)
        
    return Df

def multivariable_lag(Dataset:pd.DataFrame,
                      target:str="y",
                      range_lags:list=[1,10],
                      drop_na:bool=True):
    
    """
    The multivariable_lag function takes a Pandas DataFrame and returns a new DataFrame with the target variable 
    lagged by the number of periods specified in range_lags. The function also removes NaN values from the dataset, 
    and can be used to remove all NaN values or just those at the end of a time series. The default is to drop rows with any 
    NaNs, but this can be changed by setting drop_na=False.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be used in the function
    :param target:str: Indicate the name of the column that will be used as target
    :param range_lags:list=[1: Define the range of lags to be used
    :param 10]: Indicate the maximum lag to be used
    :param drop_na:bool=True: Drop the rows with nan values
    :return: A dataframe with the lags specified by the user
    """
    
    assert range_lags[0]>=1, "Range lags first interval value should be bigger then 1"
    assert range_lags[0]<=range_lags[1], "Range lags first interval value should be bigger then second"
    
    Df=Dataset.copy()
    Df_=Df.copy()
    Df_=slice_timestamp(Df_)
    
    lag_str,list_dfs='',[]

    for elemento in range(range_lags[0],range_lags[1]+1):
        lag_str=str(elemento)
        lag_str=target+'_lag_'+lag_str
        df_final=pd.DataFrame({'target': Df[target],
                      lag_str: Df[target].shift(elemento)
                      })
        df_final=df_final.drop('target',axis=1)
        Df_[lag_str]=df_final[lag_str]
    cols,input_cols=list(Df_.columns),list(Df_.columns)
    input_cols.remove(target)
    last_col=cols[len(cols)-1:]
    if drop_na==True:
        Df_ = Df_.dropna(axis=0, subset=last_col)
    elif drop_na==False:
        Df_[input_cols]=Df_[input_cols].apply(lambda x: x.fillna(x.mean()),axis=0)
    for col in cols:
        if col=='Date':
            Df_=Df_.set_index(['Date']) 
            break
        
    return Df_

def feature_selection_tb(Dataset:pd.DataFrame,
                         target:str="y",
                         total_vi:float=0.99,
                         algo:str="ExtraTrees",
                         estimators:int=250):
    """
    The feature_selection_tb function takes in a pandas dataframe and returns the selected columns.
    The function uses ExtraTreesRegressor to select the most important features from a dataset. 
    The user can specify how much of the total variance they want explained by their model, as well as what algorithm they would like to use for feature selection.
    
    :param Dataset:pd.DataFrame: Pass the dataset
    :param target:str=&quot;y&quot;: Specify the target variable name
    :param total_vi:float=0.99: Set the total variable importance that needs to be achieved
    :param algo:str=&quot;ExtraTrees&quot;: Select the algorithm to be used for feature selection
    :param estimators:int=250: Set the number of estimators in the randomforest and extratrees algorithms
    :return: A list of columns that are selected by the algorithm
    """
    assert total_vi>=0.5 and total_vi<=1 , "total_vi value should be in [0.5,1[ interval"
    
    Train=Dataset.copy()
    sel_cols= list(Train.columns)
    sel_cols.remove(target)
    sel_cols.append(target)
    Train=Train[sel_cols]
   
    X_train = Train.iloc[:, 0:(len(sel_cols)-1)].values
    y_train = Train.iloc[:, (len(sel_cols)-1)].values
    
    if algo=='ExtraTrees':
        fs_model = ExtraTreesRegressor(n_estimators=estimators)
        fs_model.fit(X_train, y_train)
    elif algo=="RandomForest":
        fs_model = RandomForestRegressor(n_estimators=estimators)
        fs_model.fit(X_train, y_train)
    elif algo=='GBR':
        fs_model = GradientBoostingRegressor(n_estimators=estimators)
        fs_model.fit(X_train, y_train)

    column_imp=fs_model.feature_importances_
    column_names=sel_cols.copy()
    column_names.remove(target)
    
    Feat_imp = pd.Series(column_imp)
    Columns = pd.Series(column_names)
    
    df=pd.concat([Feat_imp,Columns],axis=1)
    df = df.rename(columns={0: 'percentage',1: 'variable'})
    
    n=0.015
    va_df = df[df['percentage'] > n]
    var_importance=va_df['percentage'].sum()
    for iteration in range(0,10):
            
        if var_importance<=total_vi:
            va_df = df[df['percentage'] > n]
            n=n*0.5
            var_importance=va_df['percentage'].sum()
        elif var_importance>total_vi:
            break
    va_df = va_df.sort_values(["percentage"], ascending=False)

    sel_cols_=[]
    for rows in va_df["variable"]:
        sel_cols_.append(rows)
    sel_cols_.append(target)
    
    return sel_cols_, va_df
