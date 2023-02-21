import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import TweedieRegressor
from fbprophet import Prophet
from pmdarima.arima import auto_arima
import autokeras as ak
import xgboost
import h2o
from h2o.automl import H2OAutoML
from .tsf_model_configs import model_configurations

h_parameters=model_configurations()

def model_prediction(Train:pd.DataFrame,
                     Test:pd.DataFrame,
                     target:str="y",
                     model_configs:dict=h_parameters,
                     algo:str='RandomForest'):
    """
    The model_prediction function takes in a Train and Test dataframe, as well as the name of the target column. 
    It then trains a model using the specified algorithm (RandomForest by default) and returns predictions for 
    the test set.
    
    :param Train:pd.DataFrame: Pass the train dataframe
    :param Test:pd.DataFrame: Test the model with a different dataset
    :param target:str: Specify the column name of the target variable
    :param model_configs:dict: Pass the parameters of each model
    :param algo:str='RandomForest': Select the model to be used
    :return: The predictions of the model
    """
    sel_cols= list(Train.columns)
    sel_cols.remove(target)
    sel_cols.append(target) 
    Train=Train[sel_cols]
    Test=Test[sel_cols]   
    
    X_train = Train.iloc[:, 0:(len(sel_cols)-1)].values
    X_test = Test.iloc[:, 0:(len(sel_cols)-1)].values
    y_train = Train.iloc[:, (len(sel_cols)-1)].values
    y_test = Test.iloc[:, (len(sel_cols)-1)].values
    
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
        Test[target]=Test[target].fillna(0) ## Avoid H2O OS_Error
        Test[target]=Test[target].astype(np.float) ## Avoid H2O OS_Error
        Train_h2o,Test_h2o=h2o.H2OFrame(Train),h2o.H2OFrame(Test)
        input_cols=sel_cols.copy()
        input_cols.remove("y")
        aml_params=model_configs['H2O_AutoML']
        aml = H2OAutoML(**aml_params) 
        aml.train(x=input_cols ,y=target ,training_frame=Train_h2o)
        
        leaderboards = aml.leaderboard
        leaderboard_df= leaderboards.as_data_frame()
        id_leader_model=leaderboard_df['model_id'][0]
        h2o_leader_model=h2o.get_model(id_leader_model)
        
        pred_col = h2o_leader_model.predict(Test_h2o)
        pred_col = pred_col.asnumeric()
        y_predict= pred_col.as_data_frame()['predict']
        
    elif algo=='XGBoost':
        xg_params=model_configs['XGBoost']
        regressor_XG = xgboost.XGBRegressor(**xg_params)
        regressor_XG.fit(X_train, y_train)
        y_predict = regressor_XG.predict(X_test)
        
    elif algo=='AutoKeras':
        epochs=model_configs['AutoKeras']['epochs']
        ak_params=model_configs['AutoKeras']
        del ak_params['epochs']
        regressor_AK = ak.StructuredDataRegressor(**ak_params)
        regressor_AK.fit(X_train, y_train, epochs=epochs)
        y_predict = regressor_AK.predict(X_test)
        model_configs['AutoKeras']['epochs']=epochs
   
    return y_predict

  
