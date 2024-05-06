import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    max_error
    )

def vertical_performance(forecasts: pd.DataFrame, horizon: int):
    """
    Evaluates the performance of various forecasting models across multiple horizons.

    Parameters:
        forecasts (pd.DataFrame): A DataFrame containing actual values and predicted values
                                  for various models and windows.
        horizon (int): The number of forecasting horizons to evaluate.

    Returns:
        pd.DataFrame: A DataFrame containing performance metrics for each model, window,
                      and horizon combination.

    The function systematically computes performance metrics for each model across different
    forecasting windows and horizons. It utilizes a nested looping mechanism to handle
    various dimensions of model evaluation.
    """
    # Create a copy of the forecasts DataFrame to prevent modifying the original data.
    df = forecasts.copy()
    
    # Generate a list of integers representing each horizon number up to the specified horizon.
    horizons, results = list(range(1, horizon + 1)), [] 
    
    # Loop through each unique model in the DataFrame.
    for model in df['Model'].unique():
        
        # Loop through each unique window for the current model.
        for window in df['Window'].unique():
            # Filter the DataFrame for entries that match the current model and window.
            model_window_data = df[(df['Model'] == model) & (df['Window'] == window)]
            
            # Loop through each horizon to evaluate performance.
            for horizon in horizons:
                # Select the actual and predicted values for the current horizon.
                y_true = model_window_data[f'y_horizon_{horizon}']
                y_pred = model_window_data[f'y_forecast_horizon_{horizon}']
                
                # Store model, window, and horizon information.
                meta_data = {'Model': model,
                             'Window': window,
                             'Horizon': horizon,
                             }
                
                # Calculate performance metrics using another function (presumably defined elsewhere).
                metrics = metrics_regression(y_true, y_pred)
                
                # Append the combined metadata and metrics as a dictionary to the results list.
                results.append({**meta_data, **metrics})
    
    # Convert the list of results to a DataFrame and sort it by model, window, and horizon.
    results = pd.DataFrame(results)
    results = results.sort_values(by=['Model', 'Window', 'Horizon'])
    
    return results

def best_model(results: pd.DataFrame, metric: str = 'MAE'):
    """
   Identifies the best performing model based on a selected metric across all horizons.

   Parameters:
       results (pd.DataFrame): The DataFrame containing performance metrics for all models.
       metric (str): The performance metric to use for model comparison ('MAE', 'MAPE', 'MSE').

   Returns:
       pd.Series: Information about the best model including its name and metric value.

   This function evaluates the average performance of each model across all horizons
   and identifies the model with the best average performance based on the specified metric.
   """
    # Map the short metric name to the full metric name
    metric_mapping = {
        'MAE': 'Mean Absolute Error',
        'MAPE': 'Mean Absolute Percentage Error',
        'MSE': 'Mean Squared Error'
    }
    _metric = metric_mapping.get(metric)
        
    # Group the results DataFrame by model and horizon and calculate the mean of the specified metric.
    aggregated_metric = results.groupby(['Model', 'Horizon'])[_metric].mean().reset_index()
    
    # Calculate the mean of the specified metric for each model across all horizons.
    mean_metric_by_model = aggregated_metric.groupby('Model')[_metric].mean().reset_index()
    
    # Identify the model with the lowest mean metric value.
    best_model = mean_metric_by_model.loc[mean_metric_by_model[_metric].idxmin()]
    
    # Print and return the best model based on the lowest mean metric value.
    print('The model with the best performance was', best_model['Model'], 
          'with an (mean)', _metric, 'of', round(best_model[_metric], 4))
    
    return best_model['Model']
    
def metrics_regression(y_true, y_pred):
    """
    Calculate various regression model evaluation metrics.

    Parameters:
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns:
    pandas.DataFrame
        DataFrame containing Mean Absolute Error, Mean Squared Error, 
        Root Mean Squared Error, and R-squared metrics.
    """
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape=(mean_absolute_percentage_error(y_true, y_pred))*100
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate Root Mean Squared Error (RMSE)
    #rmse = mean_squared_error(y_true, y_pred, squared=False)
    
    # Calculate Max Error
    maxerror = max_error(y_true, y_pred)
    
    # Create a dictionary to store the metrics
    metrics = {'Mean Absolute Error': mae,
               'Mean Absolute Percentage Error': mape,
               'Mean Squared Error': mse,
               #'Root Mean Squared Error': rmse,
               'Max Error': maxerror}
    
    return metrics