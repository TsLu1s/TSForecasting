import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import TweedieRegressor
from prophet import Prophet
from pmdarima.arima import auto_arima
import xgboost
import h2o
from h2o.automl import H2OAutoML
from .tsf_model_configs import model_configurations

h_parameters=model_configurations()

def model_prediction(train:pd.DataFrame,
                     test:pd.DataFrame,
                     target:str="y",
                     model_configs:dict=h_parameters,
                     algo:str='RandomForest'):
    """
    The model_prediction function takes in a train and test dataframe, as well as the name of the target column. 
    It then trains a model using the specified algorithm (RandomForest by default) and returns predictions for 
    the test set.
    
    :param train:pd.DataFrame: Pass the train dataframe
    :param test:pd.DataFrame: test the model with a different dataset
    :param target:str: Specify the column name of the target variable
    :param model_configs:dict: Pass the parameters of each model
    :param algo:str='RandomForest': Select the model to be used
    :return: The predictions of the model
    """
    sel_cols= list(train.columns)
    sel_cols.remove(target)
    sel_cols.append(target) 
    train=train[sel_cols]
    test=test[sel_cols]   
    
    X_train = train.iloc[:, 0:(len(sel_cols)-1)].values
    X_test = test.iloc[:, 0:(len(sel_cols)-1)].values
    y_train = train.iloc[:, (len(sel_cols)-1)].values
    y_test = test.iloc[:, (len(sel_cols)-1)].values
    
    if algo=='RandomForest':
        rf_params=model_configs['RandomForest']
        regressor_RF = RandomForestRegressor(**rf_params) 
        regressor_RF.fit(X_train, y_train)
        y_predict = regressor_RF.predict(X_test)
        
    elif algo=='ExtraTrees':
        et_params=model_configs['ExtraTrees']
        regressor_ET = ExtraTreesRegressor(**et_params)
        regressor_ET.fit(X_train, y_train)
        y_predict = regressor_ET.predict(X_test)
        
    elif algo=='GBR':
        gbr_params=model_configs['GBR']
        regressor_GBR = GradientBoostingRegressor(**gbr_params)
        regressor_GBR.fit(X_train, y_train)
        y_predict = regressor_GBR.predict(X_test)
        
    elif algo=='KNN':
        knn_params=model_configs['KNN']
        regressor_KNN = KNeighborsRegressor(**knn_params)
        regressor_KNN.fit(X_train, y_train)
        y_predict = regressor_KNN.predict(X_test) 
        
    elif algo=='GeneralizedLR':
        td_params=model_configs['GeneralizedLR']
        regressor_TD = TweedieRegressor(**td_params)
        regressor_TD.fit(X_train, y_train)
        y_predict = regressor_TD.predict(X_test)
        
    elif algo=='H2O_AutoML':
        test[target]=test[target].fillna(0) ## Avoid H2O OS_Error
        test[target]=test[target].astype(float) ## Avoid H2O OS_Error
        train_h2o,test_h2o=h2o.H2OFrame(train),h2o.H2OFrame(test)
        input_cols=sel_cols.copy()
        input_cols.remove("y")
        aml_params=model_configs['H2O_AutoML']
        aml = H2OAutoML(**aml_params) 
        aml.train(x=input_cols ,y=target ,training_frame=train_h2o)
        
        leaderboards = aml.leaderboard
        leaderboard_df= leaderboards.as_data_frame()
        id_leader_model=leaderboard_df['model_id'][0]
        h2o_leader_model=h2o.get_model(id_leader_model)
        
        pred_col = h2o_leader_model.predict(test_h2o)
        pred_col = pred_col.asnumeric()
        y_predict= pred_col.as_data_frame()['predict']
        
    elif algo=='XGBoost':
        xg_params=model_configs['XGBoost']
        regressor_XG = xgboost.XGBRegressor(**xg_params)
        regressor_XG.fit(X_train, y_train)
        y_predict = regressor_XG.predict(X_test)
           
    return y_predict
