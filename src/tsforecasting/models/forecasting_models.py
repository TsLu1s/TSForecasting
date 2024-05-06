import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from autogluon.tabular import TabularPredictor
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm 

class RandomForest_Forecasting:
    """
    This class encapsulates a Random Forest regression model tailored for Forecasting tasks,
    with hyperparameters specifically configured for effective operation.
    """
    def __init__(self, n_estimators=100, random_state = 42, criterion = "squared_error", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
        """
        Initialize the RandomForestForecasting model with the specified hyperparameters.

        Parameters:
            n_estimators (int): Number of trees in the forest.
            max_depth (int or None): Maximum depth of each tree. None means nodes expand until all leaves are pure.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
            max_features (str or int): Number of features to consider when looking for the best split; "auto" uses all features.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        base_model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, criterion=self.criterion,
                                           max_depth=self.max_depth, min_samples_split=self.min_samples_split, 
                                           min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
        self.model = MultiOutputRegressor(base_model)
        
    def fit(self, X, y):
        """
        Fit the RandomForest model to the provided training data.

        Parameters:
            X (array-like): Feature dataset for training.
            y (array-like): Target values.
        """
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        """
        Predict using the fitted RandomForest model.

        Parameters:
            X (array-like): Dataset for which to make predictions.

        Returns:
            array: Predicted values.
        """
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            dict: Parameters of this estimator.
        """
        return {"n_estimators": self.n_estimators, "random_state": self.random_state,"criterion": self.criterion,
                "max_depth": self.max_depth, "min_samples_split": self.min_samples_split, 
                "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features}

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Parameters:
            parameters (dict): Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        """
        Verify that the model has been fitted.

        Raises:
            AssertionError: If the model is not fitted.
        """
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."
        
class ExtraTrees_Forecasting:
    """
    This class encapsulates an Extra Trees regression model optimized for Forecasting tasks,
    with hyperparameters specifically configured to effectively handle complex data structures.
    """
    def __init__(self, n_estimators=100, random_state = 42, criterion = "squared_error", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
        """
        Initialize the ExtraTreesForecasting model with the specified hyperparameters.

        Parameters:
            n_estimators (int): The number of trees in the forest.
            max_depth (int or None): The maximum depth of the tree. If None, then nodes are expanded until all leaves contain less than min_samples_split samples.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            max_features (str or int): The number of features to consider when looking for the best split.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        base_model = ExtraTreesRegressor(n_estimators=self.n_estimators, random_state=self.random_state, criterion=self.criterion,
                                         max_depth=self.max_depth, min_samples_split=self.min_samples_split, 
                                         min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
        self.model = MultiOutputRegressor(base_model)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "random_state": self.random_state, "criterion": self.criterion,
                "max_depth": self.max_depth, "min_samples_split": self.min_samples_split, 
                "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."

class GBR_Forecasting:
    """
    This class encapsulates a Gradient Boosting regression model designed for Forecasting, 
    integrating advanced configurations to handle various data anomalies and patterns.
    """
    def __init__(self, n_estimators=100, criterion = "friedman_mse", learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, loss = 'squared_error'):
        """
        Initialize the GBRForecasting model with the specified hyperparameters.

        Parameters:
            n_estimators (int): The number of boosting stages to be run.
            learning_rate (float): Rate at which the contribution of each tree is shrunk.
            max_depth (int): Maximum depth of the individual regression estimators.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss
        base_model = GradientBoostingRegressor(n_estimators=self.n_estimators, criterion=self.criterion, learning_rate=self.learning_rate,
                                               max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                               min_samples_leaf=self.min_samples_leaf, loss=self.loss)
        self.model = MultiOutputRegressor(base_model)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "criterion": self.criterion, "learning_rate": self.learning_rate,
                "max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf , 'loss': self.loss}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."

class KNN_Forecasting:
    """
    This class wraps a K-Nearest Neighbors regressor for use in Forecasting tasks.
    """
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2):
        """
        Initialize the KNNForecasting model with the specified hyperparameters.

        Parameters:
            n_neighbors (int): Number of neighbors to use for kneighbors queries.
            weights (str): Weight function used in prediction. Possible values: 'uniform', 'distance'.
            algorithm (str): Algorithm used to compute the nearest neighbors. Can be 'ball_tree', 'kd_tree', or 'auto'.
            leaf_size (int): Leaf size passed to BallTree or KDTree.
            p (int): Power parameter for the Minkowski metric.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        base_model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights, 
                                         algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p)
        self.model = MultiOutputRegressor(base_model)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors, "weights": self.weights, 
                "algorithm": self.algorithm, "leaf_size": self.leaf_size, "p": self.p}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."
        
class GeneralizedLR_Forecasting:
    """
    Generalized Linear Regression model using TweedieRegressor for handling multiple regression tasks with 
    MultiOutputRegressor, allowing the handling of multiple forecasting horizon targets.
    """
    def __init__(self, power=1, alpha=0.5, link='log', fit_intercept=True, max_iter=100, warm_start=False, verbose=0):
        """
        Initialize the GeneralizedLR model with the specified hyperparameters.

        Parameters:
            power (float): The power parameter for the Tweedie distribution (0=Normal, 1=Poisson, (1,2)=Compound Poisson-Gamma).
            alpha (float): Constant that multiplies the penalty terms (L1 and L2 regularization).
            link (str): The link function to use (identity, log, inverse_power).
            fit_intercept (bool): Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
            max_iter (int): Maximum number of iterations for the solver.
            warm_start (bool): Whether to reuse the solution of the previous call to fit as initialization.
            verbose (int): The verbosity level.
        """
        self.power = power
        self.alpha = alpha
        self.link = link
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.verbose = verbose
        
        base_model = TweedieRegressor(power=self.power, alpha=self.alpha, link=self.link, fit_intercept=self.fit_intercept,
                                      max_iter=self.max_iter, warm_start=self.warm_start)
        self.model = MultiOutputRegressor(base_model)

    def fit(self, X, y):
        """
        Fit the Generalized Linear Model to the provided training data.

        Parameters:
            X (array-like): Feature dataset for training.
            y (array-like): Target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict using the fitted Generalized Linear Model.

        Parameters:
            X (array-like): Dataset for which to make predictions.

        Returns:
            array: Predicted values.
        """
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns:
            dict: Parameters of this estimator.
        """
        return self.model.get_params(deep)

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Parameters:
            parameters (dict): Estimator parameters.
        """
        self.model.set_params(**parameters)        


class XGBoost_Forecasting:
    """
    This class encapsulates an XGBoost regression model tailored for Forecasting,
    leveraging powerful gradient boosting techniques to handle various types of data with high efficiency.
    """
    def __init__(self, n_estimators=100, objective = 'reg:squarederror', learning_rate=0.1, max_depth=3, 
                 reg_lambda=1, reg_alpha=0, subsample=1, colsample_bytree=1):
        """
        Initialize the XGBoostForecasting model with the specified hyperparameters.

        Parameters:
            n_estimators (int): Number of gradient boosted trees. Equivalent to the number of boosting rounds.
            learning_rate (float): Step size shrinkage used to prevent overfitting. Range is [0,1].
            max_depth (int): Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
            reg_lambda (float): L2 regularization term on weights. Increasing this value will make model more conservative.
            reg_alpha (float): L1 regularization term on weights. Increasing this value will make model more conservative.
            subsample (float): Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.
            colsample_bytree (float): Subsample ratio of columns when constructing each tree.
        """
        self.n_estimators = n_estimators
        self.objective = objective
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        base_model = xgb.XGBRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                      max_depth=self.max_depth, reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha,
                                      subsample=self.subsample, colsample_bytree=self.colsample_bytree, verbosity=0)
        self.model = MultiOutputRegressor(base_model)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators, "objective": self.objective, "learning_rate": self.learning_rate, "max_depth": self.max_depth,
            "reg_lambda": self.reg_lambda, "reg_alpha": self.reg_alpha, "subsample": self.subsample, 
            "colsample_bytree": self.colsample_bytree
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."

class CatBoost_Forecasting:
    """
   This class encapsulates a CatBoost regression model, ideal for Forecasting with its robust handling of categorical data,
   and efficient processing capabilities that minimize overfitting while maximizing predictive performance.
   """
    def __init__(self, iterations=100, loss_function = 'RMSE', depth=8, learning_rate=0.1, l2_leaf_reg=3, 
                 border_count=254, subsample=1):
        """
        Initialize the CatBoostForecasting model with the specified hyperparameters.

        Parameters:
            iterations (int): The maximum number of trees that can be built when solving machine learning problems.
            depth (int): Depth of the tree. A deep tree can model more complex relationships by adding more splits; it also risks overfitting.
            learning_rate (float): The learning rate used in updating the model as it attempts to minimize the loss function.
            l2_leaf_reg (float): Coefficient at the L2 regularization term of the cost function, which controls the trade-off between achieving lower training error and reducing model complexity to avoid overfitting.
            border_count (int): The number of splits for numerical features used to find the optimal cut points.
            subsample (float): The subsample ratio of the training instance. Setting it lower can prevent overfitting but may raise the variance of the model.
        """
        self.iterations = iterations
        self.loss_function = loss_function
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.subsample = subsample
        base_model = CatBoostRegressor(
            iterations=self.iterations,
            loss_function = self.loss_function,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            subsample=self.subsample,
            save_snapshot=False,
            verbose=False
        )
        self.model = MultiOutputRegressor(base_model)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            "iterations": self.iterations, "depth": self.depth, "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg, "border_count": self.border_count, "subsample": self.subsample
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."

class LightGBM_Forecasting:
    """
    This class encapsulates a LightGBM regression model, well-suited for Forecasting tasks with large datasets,
    utilizing light gradient boosting mechanism that is resource-efficient yet delivers high performance.
    """

    def __init__(self, boosting_type='gbdt', objective='regression', metric = 'mse', num_leaves=31, max_depth=-1, learning_rate=0.1, 
                 n_estimators=100, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, 
                 reg_lambda=0.1, verbose=-1, verbosity=-1, force_col_wise=True, min_data_in_leaf=20):
        """
        Initialize the LightGBMForecasting model with the specified hyperparameters.

        Parameters:
            boosting_type (str): Type of boosting to perform.
            objective (str): The optimization objective of the model.
            num_leaves (int): Maximum number of leaves in one tree.
            max_depth (int): Maximum tree depth for base learners.
            learning_rate (float): Boosting learning rate.
            n_estimators (int): Number of boosted trees to fit.
            feature_fraction (float): Fraction of features to be used in each iteration.
            bagging_fraction (float): Fraction of data to be used for each tree.
            bagging_freq (int): Frequency of bagging.
            reg_alpha (float): L1 regularization term.
            reg_lambda (float): L2 regularization term.
            verbose (int): Verbosity for logging.
            verbosity (int) : Verbosity for logging.
            force_col_wise (bool): Force col-wise histogram building.
            min_data_in_leaf (int): Minimum number of samples in one leaf.
        """
        self.boosting_type = boosting_type
        self.objective = objective
        self.metric = metric
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        self.verbosity = verbosity
        self.force_col_wise = force_col_wise
        self.min_data_in_leaf = min_data_in_leaf
        self.model = MultiOutputRegressor(lgb.LGBMRegressor(boosting_type=self.boosting_type, objective=self.objective,
                                                            num_leaves=self.num_leaves, max_depth=self.max_depth,
                                                            learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                                                            feature_fraction=self.feature_fraction, bagging_fraction=self.bagging_fraction,
                                                            bagging_freq=self.bagging_freq, reg_alpha=self.reg_alpha,
                                                            reg_lambda=self.reg_lambda, verbose=self.verbose, verbosity=self.verbosity,
                                                            force_col_wise=self.force_col_wise, min_data_in_leaf=self.min_data_in_leaf))
    def fit(self, X, y):
        """
        Fit the LightGBM model to the provided training data using the specified parameters.

        Parameters:
            X (array-like): Feature dataset for training.
            y (array-like): Target values.
        """
        self.model.fit(X, y)
        self.is_fitted_ = True
        
    def predict(self, X):
        """
        Predict using the fitted LightGBM model.

        Parameters:
            X (array-like): Dataset for which to make predictions.

        Returns:
            array: Predicted values.
        """
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return self.model.get_params(deep)

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        """
        for key, value in parameters.items():
            if key in self.params:
                setattr(self, key, value)
                self.params[key] = value
        self.is_fitted_ = False  # Invalidate the model fitting
        return self

    def check_is_fitted(self):
        """
        Verify that the model has been fitted.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise AttributeError("This LightGBMForecasting instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

class AutoGluon_Forecasting:
    """
    This class encapsulates an AutoGluon TabularPredictor for regression tasks,
    designed to handle multi-target forecasting by fitting individual models for each target.
    """
    def __init__(self, labels, eval_metric:str='mean_squared_error', verbosity:int=0, presets:str='good_quality'):
        """
        Initialize the AutoGluonForecasting model for multiple targets.

        Parameters:
            labels (list of str): Names of the target variable columns in the data.
            eval_metric (str): Evaluation metric to be used for regression tasks.
            verbosity (int): The verbosity levels range from 0 (silent) to 3 (detailed output).
            presets (str): Preset configurations that trade off between various aspects of training speed, predictive performance, and memory footprint.
        """
        self.labels = labels
        self.eval_metric = eval_metric
        self.verbosity = verbosity
        self.presets = presets
        self.predictors = {}  # Stores a predictor for each label

    def fit(self, train_data, time_limit:int=15, save_space:str=False):
        """
        Fit an AutoGluon model for each target label using the provided training data.
        """
        for label in tqdm(self.labels, desc="Fitting AutoGluon Multivariate Forecasting", ncols = 80):
            #print(f"Training model for target: {label}")
            predictor = TabularPredictor(label=label, eval_metric=self.eval_metric, verbosity=self.verbosity).fit(
                train_data, presets= self.presets, time_limit=time_limit, save_space=save_space)
            self.predictors[label] = predictor
        self.is_fitted_ = True

    def predict(self, test_data):
        """
        Predict using the fitted AutoGluon models for each target.
        """
        self.check_is_fitted()
        predictions = pd.DataFrame(index=test_data.index)
        for label, predictor in self.predictors.items():
            predictions[label] = predictor.predict(test_data)
        return predictions

    def check_is_fitted(self):
        """
        Verify that the models have been fitted.
        """
        if not self.predictors:
            raise AttributeError("This AutoGluonForecasting instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
            