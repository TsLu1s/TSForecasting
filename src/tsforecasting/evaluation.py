import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    explained_variance_score,
    max_error
    )

def vertical_performance(dataset: pd.DataFrame,
                         forecast_length: int):
    """
    Computes performance metrics for each forecast horizon in a vertical manner.

    Args:
        dataset (pd.DataFrame): DataFrame containing 'Window', 'y_true', and 'y_pred' columns.
        forecast_length (int): Maximum forecast length.

    Returns:
        pd.DataFrame: DataFrame with performance metrics for each forecast horizon.
    """
    X=dataset.copy()

    ahead_forecast,l_steps,l_dfs=list(range(1,forecast_length+1)),(list(dict.fromkeys(X['Window'].tolist()))),[]

    for step in l_steps:
        X_=X[X['Window']==step]
        X_['V_Window']=ahead_forecast
        l_dfs.append(X_)
    df_v=pd.concat(l_dfs)
    
    a_steps,l_metrics=(list(dict.fromkeys(df_v['V_Window'].tolist()))),[]

    for e in a_steps:
        x=df_v.loc[df_v['V_Window'] == e]
        v_metrics=pd.DataFrame(metrics_regression(x['y_true'],x['y_pred']),index=[0])
        v_metrics[['Forecast_Length']]=e
        l_metrics.append(v_metrics)

    f_metrics=pd.concat(l_metrics)

    return f_metrics

def select_model(dataset: pd.DataFrame,
                 metric: str = 'MAE'):
    """
    Selects the best-performing model based on a specified metric.

    Args:
        dataset (pd.DataFrame): DataFrame containing 'Model' and the specified metric columns.
        metric (str, optional): Metric to use for model selection ('MAE', 'MAPE', 'MSE'). Defaults to 'MAE'.

    Returns:
        str: Best-performing model.
    """
    m_perf={}
    
    if metric == 'MAE': metric_='Mean Absolute Error'
    elif metric == 'MAPE': metric_='Mean Absolute Percentage Error'
    elif metric == 'MSE': metric_='Mean Squared Error'
    
    l_models=(list(dict.fromkeys(dataset['Model'].tolist())))

    for m in l_models:
        X=dataset.copy()
        X=X.loc[X['Model'] == m]
        m_model=round(X[metric_].mean(),4)
        m_perf[m]=m_model
        
    print('Models Predictive Performance:', m_perf)
    b_model=min(m_perf, key=m_perf.get)
    value=dict(sorted(m_perf.items(), key=lambda item: item[1]))[min(m_perf, key=m_perf.get)]
    if len(l_models)>1:
        print('The model with best performance was', b_model, 
              'with an (mean)', metric_, 'of', value )
    return b_model

def metrics_regression(y_true, y_pred):
    """
    Computes regression metrics.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        dict: Dictionary containing regression metrics.
    """
    mae=mean_absolute_error(y_true, y_pred)
    mape=(mean_absolute_percentage_error(y_true, y_pred))*100
    mse=mean_squared_error(y_true, y_pred)
    evs=explained_variance_score(y_true, y_pred)
    error=max_error(y_true, y_pred)
    metrics={'Mean Absolute Error': mae, 
             'Mean Absolute Percentage Error': mape,
             'Mean Squared Error': mse,
             'Explained Variance Score': evs,
             'Max Error': error}
    
    return metrics