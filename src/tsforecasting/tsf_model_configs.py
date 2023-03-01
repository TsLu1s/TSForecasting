def model_configurations():
    
    model_configs ={'RandomForest':{'n_estimators':250,'random_state':42,'criterion':"squared_error",
                       'max_depth':None,'max_features':"auto"},
                   'ExtraTrees':{'n_estimators':250,'random_state':42,'criterion':"squared_error",
                       'max_depth':None,'max_features':"auto"}, 
                   'GBR':{'n_estimators':250,'learning_rate':0.1,'criterion':"friedman_mse",
                        'max_depth':3,'min_samples_split':5,'learning_rate':0.01,'loss':'ls'},
                   'KNN':{'n_neighbors': 3,'weights':"uniform",'algorithm':"auto",'metric_params':None},
                   'GeneralizedLR':{'power':1,'alpha':0.5,'link':'log','fit_intercept':True,
                        'max_iter':100,'warm_start':False,'verbose':0},
                   'XGBoost':{'objective':'reg:squarederror','n_estimators':1000,'nthread':24},
                   'H2O_AutoML':{'max_models':50,'nfolds':0,'seed':1,'max_runtime_secs':30,
                        'sort_metric':'AUTO','exclude_algos':['GBM','DeepLearning']},
                   'AutoArima':{'start_p':0, 'd':1, 'start_q':0,'max_p':3, 'max_d':3, 'max_q':3,
                        'start_P':0, 'D':1, 'start_Q':0, 'max_P':3, 'max_D':3,'max_Q':3,
                        'm':1,'trace':True, 'seasonal':True,'random_state':20,'n_fits':10},
                   'Prophet':{'growth':'linear','changepoint_range':0.8,'yearly_seasonality':'auto',
                        'weekly_seasonality':'auto','daily_seasonality':'auto'},
                    }
    
    return model_configs






















